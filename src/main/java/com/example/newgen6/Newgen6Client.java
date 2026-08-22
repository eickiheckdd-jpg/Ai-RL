package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.util.Identifier;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.Hand;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;

import org.lwjgl.glfw.GLFW;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Predicate;

public final class Newgen6Client implements ClientModInitializer {

    private static final String MOD_ID = "newgen6";
    private static final KeyBinding.Category NEWGEN6_CATEGORY = KeyBinding.Category.create(Identifier.of(MOD_ID, "general"));

    private static Path DATA_DIR;
    private static Path MODEL_FILE;
    private static KeyBinding MASTER_KEY;
    private static KeyBinding UI_KEY;

    private static boolean enabled = false;
    private static boolean uiEnabled = false;

    // --- HUD Stats ---
    private static String currentActionName = "None";
    private static double currentLoss = 0.0;
    private static double avgReward = 0.0;
    private static int actionsTaken = 0;

    // --- AI Parameters ---
    private static final int MAX_FRAME_INPUTS = 32;
    private static final int HISTORY_FRAMES = 50;
    private static final int TOTAL_INPUTS = MAX_FRAME_INPUTS * HISTORY_FRAMES;
    
    private static final int MAX_ACTIONS = 32; 
    private static final List<AIAction> ACTION_REGISTRY = new ArrayList<>();
    
    // Action Repeat / Frame Skipping
    private static final int ACTION_REPEAT = 4;
    private static int actionTimer = 0;

    private static final Random RANDOM = new Random();
    private static NeuralPolicy POLICY;

    private static final double[] HISTORY = new double[TOTAL_INPUTS];
    private static boolean historyInitialized = false;

    private static Observation previousObservation;
    private static int previousAction = -1;

    @Override
    public void onInitializeClient() {
        MinecraftClient client = MinecraftClient.getInstance();
        DATA_DIR = client.runDirectory.toPath().resolve(MOD_ID);
        MODEL_FILE = DATA_DIR.resolve("model.bin");

        try { Files.createDirectories(DATA_DIR); } catch (IOException e) { e.printStackTrace(); }

        registerBaseActions();
        POLICY = new NeuralPolicy(TOTAL_INPUTS, 128, 64, MAX_ACTIONS);
        loadModel();

        MASTER_KEY = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.newgen6.master", GLFW.GLFW_KEY_C, NEWGEN6_CATEGORY));
        UI_KEY = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.newgen6.ui", GLFW.GLFW_KEY_X, NEWGEN6_CATEGORY));
        
        ClientTickEvents.END_CLIENT_TICK.register(Newgen6Client::tick);
        
        // Render the Neural HUD
        HudRenderCallback.EVENT.register((drawContext, tickDelta) -> {
            if (!uiEnabled || client.textRenderer == null) return;
            int y = 10;
            drawContext.drawText(client.textRenderer, "Newgen6 A2C Engine", 10, y, 0x00FF00, false); y += 12;
            drawContext.drawText(client.textRenderer, "Mode: " + (enabled ? "ACTIVE" : "IDLE"), 10, y, enabled ? 0x00FF00 : 0xFF0000, false); y += 12;
            drawContext.drawText(client.textRenderer, "Current Action: " + currentActionName, 10, y, 0xFFFFFF, false); y += 12;
            drawContext.drawText(client.textRenderer, String.format("Moving Avg Reward: %.3f", avgReward), 10, y, 0xFFFFFF, false); y += 12;
            drawContext.drawText(client.textRenderer, String.format("A2C Advantage (Loss): %.3f", currentLoss), 10, y, 0xFFFFFF, false); y += 12;
            drawContext.drawText(client.textRenderer, "Total Steps: " + actionsTaken, 10, y, 0xAAAAAA, false);
        });
    }

    private static void registerBaseActions() {
        float[] yawSteps = {-4f, 4f, 0f, 0f, -2f, 2f, -1f, 1f};
        float[] pitchSteps = {0f, 0f, -3f, 3f, -1.5f, 1.5f, -0.5f, 0.5f};

        for (int i = 0; i < yawSteps.length; i++) {
            final float dy = yawSteps[i];
            final float dp = pitchSteps[i];
            ACTION_REGISTRY.add(new AIAction("Aim_" + i, client -> {
                if (client.player != null) {
                    client.player.setYaw(client.player.getYaw() + dy);
                    client.player.setPitch(clampPitch(client.player.getPitch() + dp));
                }
            }, client -> true));
        }

        ACTION_REGISTRY.add(new AIAction("Attack", client -> {
            if (client.interactionManager != null && client.player != null && client.targetedEntity instanceof PlayerEntity target) {
                client.interactionManager.attackEntity(client.player, target);
                client.player.swingHand(Hand.MAIN_HAND);
            }
        }, client -> client.player != null && client.player.getAttackCooldownProgress(0.5f) >= 0.9f && client.targetedEntity instanceof PlayerEntity));
    }
    private static void tick(MinecraftClient client) {
        if (client.player == null || client.world == null) return;

        while (MASTER_KEY.wasPressed()) {
            enabled = !enabled;
            if (enabled) {
                resetSession();
                System.out.println("[Newgen6] ENABLED - A2C Active");
            } else {
                System.out.println("[Newgen6] DISABLED");
                client.options.attackKey.setPressed(false);
                saveModel();
                resetSession();
            }
        }
        
        while (UI_KEY.wasPressed()) {
            uiEnabled = !uiEnabled;
        }

        if (!enabled) return;

        PlayerEntity enemy = findNearestOpponent(client.player);
        if (client.player.isDead() || (enemy != null && enemy.isDead())) {
            double terminalReward = client.player.isDead() ? -50.0 : 50.0;
            if (previousObservation != null) {
                Observation terminalState = observe(client.player, enemy != null ? enemy : client.player);
                if (terminalState != null) {
                    double vNext = POLICY.forward(terminalState.values).value;
                    CompletableFuture.runAsync(() -> POLICY.train(previousObservation.values, previousAction, terminalReward, vNext, true, getActionMasks(client)));
                }
            }
            resetSession();
            return;
        }

        // Action Repeat Check
        if (actionTimer > 0 && previousAction != -1) {
            if (previousAction < ACTION_REGISTRY.size()) {
                ACTION_REGISTRY.get(previousAction).executor.accept(client);
            }
            actionTimer--;
            return;
        }

        Observation frame = observe(client.player, enemy);
        if (frame == null) return;

        if (!historyInitialized) {
            for (int i = 0; i < HISTORY_FRAMES; i++) {
                System.arraycopy(frame.values, 0, HISTORY, i * MAX_FRAME_INPUTS, MAX_FRAME_INPUTS);
            }
            historyInitialized = true;
        }

        System.arraycopy(HISTORY, MAX_FRAME_INPUTS, HISTORY, 0, TOTAL_INPUTS - MAX_FRAME_INPUTS);
        System.arraycopy(frame.values, 0, HISTORY, TOTAL_INPUTS - MAX_FRAME_INPUTS, MAX_FRAME_INPUTS);
        
        runAI(client, new Observation(HISTORY));
        actionTimer = ACTION_REPEAT; // Lock in this choice for the next few ticks
        
        if (client.player.age % 1200 == 0) saveModel();
    }

    private static void resetSession() {
        previousObservation = null;
        previousAction = -1;
        actionTimer = 0;
        Arrays.fill(HISTORY, 0.0);
        historyInitialized = false;
    }
    private static double raycastDistance(ClientPlayerEntity player, Vec3d offset) {
        Vec3d start = player.getEyePos();
        Vec3d end = start.add(offset.multiply(5.0)); // 5 block sight range
        RaycastContext context = new RaycastContext(start, end, RaycastContext.ShapeType.COLLIDER, RaycastContext.FluidHandling.NONE, player);
        net.minecraft.util.hit.BlockHitResult result = player.getEntityWorld().raycast(context);
        if (result.getType() == net.minecraft.util.hit.HitResult.Type.MISS) return 1.0;
        return clamp(start.distanceTo(result.getPos()) / 5.0);
    }

    private static Observation observe(ClientPlayerEntity player, PlayerEntity enemy) {
        if (enemy == null) return null;

        Vec3d eyePos = player.getEyePos();
        Vec3d targetPos = enemy.getEyePos();
        Vec3d diff = targetPos.subtract(eyePos);
        double distance = diff.length();

        var pv = player.getVelocity();
        var ev = enemy.getVelocity();

        Vec3d look = player.getRotationVector();
        Vec3d right = look.crossProduct(new Vec3d(0, 1, 0)).normalize();

        double[] x = new double[MAX_FRAME_INPUTS];
        x[0] = clamp(distance / 10.0);
        x[1] = clamp(diff.x / 10.0);
        x[2] = clamp(diff.y / 5.0);
        x[3] = clamp(diff.z / 10.0);
        x[4] = clamp((ev.x - pv.x) / 2.0);
        x[5] = clamp((ev.y - pv.y) / 2.0);
        x[6] = clamp((ev.z - pv.z) / 2.0);
        x[7] = clamp(player.getHealth() / player.getMaxHealth());
        x[8] = clamp(enemy.getHealth() / enemy.getMaxHealth());
        x[9] = player.getAttackCooldownProgress(0.0f);
        x[10] = player.isSprinting() ? 1.0 : -1.0;
        x[11] = player.isOnGround() ? 1.0 : -1.0;
        
        Vec3d idealLook = diff.normalize();
        x[12] = look.dotProduct(idealLook); // 3D aim alignment

        // Advanced Sensors (Whiskers & Enemy Threat)
        x[13] = raycastDistance(player, look); // Front Whisker
        x[14] = raycastDistance(player, look.multiply(-1)); // Back Whisker
        x[15] = raycastDistance(player, right.multiply(-1)); // Left Whisker
        x[16] = raycastDistance(player, right); // Right Whisker

        Vec3d enemyLook = enemy.getRotationVector();
        x[17] = enemyLook.dotProduct(diff.multiply(-1).normalize()); // 1.0 if enemy is looking directly at us

        return new Observation(x);
    }

    private static PlayerEntity findNearestOpponent(ClientPlayerEntity player) {
        PlayerEntity best = null;
        double bestDistanceSq = 14.0 * 14.0;
        for (PlayerEntity other : player.getEntityWorld().getPlayers()) {
            if (other == player || other.isDead()) continue;
            double d = player.squaredDistanceTo(other);
            if (d < bestDistanceSq) { bestDistanceSq = d; best = other; }
        }
        return best;
    }
    private static boolean[] getActionMasks(MinecraftClient client) {
        boolean[] masks = new boolean[MAX_ACTIONS];
        for (int i = 0; i < MAX_ACTIONS; i++) masks[i] = (i < ACTION_REGISTRY.size()) && ACTION_REGISTRY.get(i).condition.test(client);
        return masks;
    }

    private static void runAI(MinecraftClient client, Observation current) {
        boolean[] masks = getActionMasks(client);
        int action = POLICY.chooseAction(current.values, masks, 1.0);
        
        currentActionName = (action < ACTION_REGISTRY.size()) ? ACTION_REGISTRY.get(action).name : "Unknown";
        if (action < ACTION_REGISTRY.size()) ACTION_REGISTRY.get(action).executor.accept(client);

        PlayerEntity enemy = findNearestOpponent(client.player);
        Observation nextFrame = observe(client.player, enemy);
        if (nextFrame == null) return;
        
        double[] nextHistory = HISTORY.clone();
        System.arraycopy(nextHistory, MAX_FRAME_INPUTS, nextHistory, 0, TOTAL_INPUTS - MAX_FRAME_INPUTS);
        System.arraycopy(nextFrame.values, 0, nextHistory, TOTAL_INPUTS - MAX_FRAME_INPUTS, MAX_FRAME_INPUTS);
        Observation nextState = new Observation(nextHistory);

        if (previousObservation != null) {
            double reward = calculateUnifiedReward(previousObservation, nextState);
            avgReward = (avgReward * 0.99) + (reward * 0.01);
            
            double[] oldStateArray = previousObservation.values.clone();
            int oldAction = previousAction;
            double vNext = POLICY.forward(nextState.values).value;
            CompletableFuture.runAsync(() -> POLICY.train(oldStateArray, oldAction, reward, vNext, false, masks));
            actionsTaken++;
        }

        previousObservation = nextState;
        previousAction = action;
    }
    private static double calculateUnifiedReward(Observation oldState, Observation newState) {
        final int last = TOTAL_INPUTS - MAX_FRAME_INPUTS;
        
        double damageDealt = Math.max(0.0, oldState.values[last + 8] - newState.values[last + 8]);
        double damageTaken = Math.max(0.0, oldState.values[last + 7] - newState.values[last + 7]);
        double reward = (damageDealt * 35.0) - (damageTaken * 40.0);

        double realDistance = newState.values[last + 0] * 10.0;
        double rangeError = Math.abs(realDistance - 3.0);
        reward -= (rangeError / 5.0);

        double currentAimAlignment = newState.values[last + 12];
        reward += (currentAimAlignment > 0.98) ? 0.5 : (currentAimAlignment * 0.1); 

        return Math.max(-50.0, Math.min(50.0, reward));
    }

    private record AIAction(String name, Consumer<MinecraftClient> executor, Predicate<MinecraftClient> condition) {}
    private record Forward(double[] h1, double[] h2, double[] logits, double value) {}
    private record Observation(double[] values) { private Observation(double[] values) { this.values = Arrays.copyOf(values, values.length); } }
    private static final class NeuralPolicy {
        private final int input, hidden1, hidden2, numActions;
        private final double[][] w1, w2, w3, wV;
        private final double[] b1, b2, b3, bV;
        private final double[][] mw1, mw2, mw3, mwV, vw1, vw2, vw3, vwV;
        private final double[] mb1, mb2, mb3, mbV, vb1, vb2, vb3, vbV;
        private int adamStep = 0;

        NeuralPolicy(int input, int hidden1, int hidden2, int numActions) {
            this.input = input; this.hidden1 = hidden1; this.hidden2 = hidden2; this.numActions = numActions;
            w1 = new double[input][hidden1]; w2 = new double[hidden1][hidden2]; 
            w3 = new double[hidden2][numActions]; wV = new double[hidden2][1];
            b1 = new double[hidden1]; b2 = new double[hidden2]; b3 = new double[numActions]; bV = new double[1];

            mw1 = new double[input][hidden1]; mw2 = new double[hidden1][hidden2]; mw3 = new double[hidden2][numActions]; mwV = new double[hidden2][1];
            vw1 = new double[input][hidden1]; vw2 = new double[hidden1][hidden2]; vw3 = new double[hidden2][numActions]; vwV = new double[hidden2][1];
            mb1 = new double[hidden1]; mb2 = new double[hidden2]; mb3 = new double[numActions]; mbV = new double[1];
            vb1 = new double[hidden1]; vb2 = new double[hidden2]; vb3 = new double[numActions]; vbV = new double[1];

            randomize(w1, input); randomize(w2, hidden1); randomize(w3, hidden2); randomize(wV, hidden2);
        }

        private void randomize(double[][] matrix, int fanIn) {
            double scale = Math.sqrt(2.0 / fanIn);
            for (int i = 0; i < matrix.length; i++) for (int j = 0; j < matrix[i].length; j++) matrix[i][j] = RANDOM.nextGaussian() * scale;
        }

        Forward forward(double[] x) {
            double[] h1 = new double[hidden1]; double[] h2 = new double[hidden2]; double[] logits = new double[numActions];
            for (int j = 0; j < hidden1; j++) { double s = b1[j]; for (int i = 0; i < input; i++) s += x[i] * w1[i][j]; h1[j] = Math.max(0, s); }
            for (int j = 0; j < hidden2; j++) { double s = b2[j]; for (int i = 0; i < hidden1; i++) s += h1[i] * w2[i][j]; h2[j] = Math.max(0, s); }
            for (int j = 0; j < numActions; j++) { double s = b3[j]; for (int i = 0; i < hidden2; i++) s += h2[i] * w3[i][j]; logits[j] = s; }
            double val = bV[0]; for (int i = 0; i < hidden2; i++) val += h2[i] * wV[i][0];
            return new Forward(h1, h2, logits, val);
        }

        int chooseAction(double[] x, boolean[] masks, double temperature) {
            Forward f = forward(x);
            for (int i = 0; i < numActions; i++) if (!masks[i]) f.logits[i] = -1e9;
            double[] p = softmax(f.logits, temperature);
            double r = RANDOM.nextDouble(), cum = 0.0;
            for (int i = 0; i < p.length; i++) { cum += p[i]; if (r <= cum) return i; }
            return 0;
        }
        void train(double[] x, int action, double reward, double vNext, boolean terminal, boolean[] masks) {
            if (action < 0 || action >= numActions) return;
            adamStep++;
            Forward f = forward(x);
            for (int i = 0; i < numActions; i++) if (!masks[i]) f.logits[i] = -1e9;
            double[] p = softmax(f.logits, 1.0);
            
            double target = reward + (terminal ? 0.0 : 0.99 * vNext);
            double advantage = target - f.value;
            currentLoss = advantage;
            
            double[] dLogits = new double[numActions];
            for (int i = 0; i < numActions; i++) {
                if (!masks[i]) continue;
                dLogits[i] = (-p[i] * advantage) + (0.01 * p[i] * Math.log(p[i] + 1e-8));
            }
            dLogits[action] += advantage;
            double dValue = -advantage;

            double[][] dw3 = new double[hidden2][numActions]; double[] db3 = new double[numActions];
            for (int i = 0; i < hidden2; i++) for (int j = 0; j < numActions; j++) dw3[i][j] = dLogits[j] * f.h2[i];
            System.arraycopy(dLogits, 0, db3, 0, numActions);

            double[][] dwV = new double[hidden2][1]; double[] dbV = new double[]{dValue};
            for (int i = 0; i < hidden2; i++) dwV[i][0] = dValue * f.h2[i];

            double[] dh2 = new double[hidden2];
            for (int i = 0; i < hidden2; i++) {
                double s = dValue * wV[i][0];
                for (int j = 0; j < numActions; j++) s += dLogits[j] * w3[i][j];
                dh2[i] = (f.h2[i] > 0) ? s : 0;
            }

            double[][] dw2 = new double[hidden1][hidden2]; double[] db2 = new double[hidden2];
            for (int i = 0; i < hidden1; i++) for (int j = 0; j < hidden2; j++) dw2[i][j] = dh2[j] * f.h1[i];
            System.arraycopy(dh2, 0, db2, 0, hidden2);

            double[] dh1 = new double[hidden1];
            for (int i = 0; i < hidden1; i++) {
                double s = 0; for (int j = 0; j < hidden2; j++) s += dh2[j] * w2[i][j];
                dh1[i] = (f.h1[i] > 0) ? s : 0;
            }

            double[][] dw1 = new double[input][hidden1]; double[] db1 = new double[hidden1];
            for (int i = 0; i < input; i++) for (int j = 0; j < hidden1; j++) dw1[i][j] = dh1[j] * x[i];
            System.arraycopy(dh1, 0, db1, 0, hidden1);

            updateAdamMatrix(w3, mw3, vw3, dw3); updateAdamArray(b3, mb3, vb3, db3);
            updateAdamMatrix(wV, mwV, vwV, dwV); updateAdamArray(bV, mbV, vbV, dbV);
            updateAdamMatrix(w2, mw2, vw2, dw2); updateAdamArray(b2, mb2, vb2, db2);
            updateAdamMatrix(w1, mw1, vw1, dw1); updateAdamArray(b1, mb1, vb1, db1);
        }
        private void updateAdamMatrix(double[][] w, double[][] m, double[][] v, double[][] g) {
            for (int i = 0; i < w.length; i++) for (int j = 0; j < w[i].length; j++) {
                m[i][j] = 0.9 * m[i][j] + 0.1 * g[i][j]; v[i][j] = 0.999 * v[i][j] + 0.001 * g[i][j] * g[i][j];
                w[i][j] -= 0.0003 * (m[i][j] / (1 - Math.pow(0.9, adamStep))) / (Math.sqrt(v[i][j] / (1 - Math.pow(0.999, adamStep))) + 1e-8);
            }
        }
        private void updateAdamArray(double[] w, double[] m, double[] v, double[] g) {
            for (int i = 0; i < w.length; i++) {
                m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
                w[i] -= 0.0003 * (m[i] / (1 - Math.pow(0.9, adamStep))) / (Math.sqrt(v[i] / (1 - Math.pow(0.999, adamStep))) + 1e-8);
            }
        }
        private double[] softmax(double[] l, double t) {
            double[] p = new double[l.length]; double m = Arrays.stream(l).max().orElse(0), s = 0;
            for (int i = 0; i < l.length; i++) { p[i] = Math.exp((l[i] - m) / t); s += p[i]; }
            for (int i = 0; i < p.length; i++) p[i] /= s; return p;
        }
    }
    private static void saveModel() {
        if (MODEL_FILE == null) return;
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(MODEL_FILE)))) {
            out.writeInt(3);
            writeM(out, POLICY.w1); writeM(out, POLICY.w2); writeM(out, POLICY.w3); writeM(out, POLICY.wV);
            writeA(out, POLICY.b1); writeA(out, POLICY.b2); writeA(out, POLICY.b3); writeA(out, POLICY.bV);
        } catch (IOException ignored) {}
    }

    private static void loadModel() {
        if (MODEL_FILE == null || !Files.exists(MODEL_FILE)) return;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(MODEL_FILE)))) {
            if (in.readInt() != 3) throw new IOException("Old model version.");
            readM(in, POLICY.w1); readM(in, POLICY.w2); readM(in, POLICY.w3); readM(in, POLICY.wV);
            readA(in, POLICY.b1); readA(in, POLICY.b2); readA(in, POLICY.b3); readA(in, POLICY.bV);
        } catch (IOException ignored) {}
    }

    private static void writeM(DataOutputStream o, double[][] m) throws IOException { o.writeInt(m.length); for (double[] r : m) { o.writeInt(r.length); for (double v : r) o.writeDouble(v); } }
    private static void readM(DataInputStream in, double[][] m) throws IOException { int r = in.readInt(); for (int i = 0; i < r; i++) { int c = in.readInt(); for (int j = 0; j < c; j++) m[i][j] = in.readDouble(); } }
    private static void writeA(DataOutputStream o, double[] a) throws IOException { o.writeInt(a.length); for (double v : a) o.writeDouble(v); }
    private static void readA(DataInputStream in, double[] a) throws IOException { int l = in.readInt(); for (int i = 0; i < l; i++) a[i] = in.readDouble(); }
    private static double clamp(double v) { return Math.max(-1.0, Math.min(1.0, v)); }
    private static float clampPitch(float p) { return Math.max(-90.0f, Math.min(90.0f, p)); }
}
