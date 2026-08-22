package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;

import net.minecraft.client.MinecraftClient;
import java.util.Random;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.util.Identifier;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;

import org.lwjgl.glfw.GLFW;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Consumer;

/**
 * Newgen6 - Fully Soft-Coded Dynamic RL Engine (Chunk 1).
 */
public final class Newgen6Client implements ClientModInitializer {

    private static final String MOD_ID = "newgen6";

    private static final KeyBinding.Category NEWGEN6_CATEGORY =
            KeyBinding.Category.create(
                    Identifier.of("newgen6", "general")
            );

    private static Path DATA_DIR;
    private static Path MODEL_FILE;
    private static KeyBinding MASTER_KEY;

    private static boolean enabled = false;
    private static boolean aiEnabled = false;

    private static final int FRAME_INPUTS = 19;
    private static final int HISTORY_FRAMES = 50; 
    private static final int INPUTS = FRAME_INPUTS * HISTORY_FRAMES;
    
    private static final List<AIAction> ACTION_REGISTRY = new ArrayList<>();
    private static final int RESERVED_FUTURE_SLOTS = 10;

    private static final Random RANDOM = new Random();
    private static NeuralPolicy POLICY;

    private static final double LEARNING_RATE = 0.001;
    private static final double GAMMA = 0.99;
    private static final double ENTROPY_BETA = 0.01;

    private static final double[] HISTORY = new double[INPUTS];
    private static boolean historyInitialized = false;

    private static Observation previousObservation;
    private static int previousAction = -1;
    private static double accumulatedReturn = 0.0;
    private static double gammaMultiplier = 1.0;

    @Override
    public void onInitializeClient() {
        MinecraftClient client = MinecraftClient.getInstance();

        DATA_DIR = client.runDirectory.toPath().resolve("newgen6");
        MODEL_FILE = DATA_DIR.resolve("model.bin");

        try {
            Files.createDirectories(DATA_DIR);
        } catch (IOException e) {
            e.printStackTrace();
        }

        registerBaseActions();

        int totalActions = ACTION_REGISTRY.size() + RESERVED_FUTURE_SLOTS;
        POLICY = new NeuralPolicy(INPUTS, 64, 64, totalActions);

        loadModel();

        MASTER_KEY = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.master",
                        GLFW.GLFW_KEY_C,
                        NEWGEN6_CATEGORY
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(Newgen6Client::tick);

        System.out.println("[Newgen6] Loaded with " + ACTION_REGISTRY.size() + " active actions (" + totalActions + " total pooled slots).");
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
            }));
        }

        ACTION_REGISTRY.add(new AIAction("Attack", client -> {
            if (client.interactionManager != null && client.player != null) {
                if (client.targetedEntity instanceof PlayerEntity target) {
                    if (client.player.getAttackCooldownProgress(0.5f) >= 0.9f) {
                        client.interactionManager.attackEntity(client.player, target);
                        client.player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
                    }
                }
            }
        }));
    }
    private static void tick(MinecraftClient client) {
        if (client.player == null || client.world == null) {
            return;
        }

        while (MASTER_KEY.wasPressed()) {
            enabled = !enabled;
            aiEnabled = enabled;

            if (enabled) {
                previousObservation = null;
                previousAction = -1;
                accumulatedReturn = 0.0;
                gammaMultiplier = 1.0;
                Arrays.fill(HISTORY, 0.0);
                historyInitialized = false;

                System.out.println("[Newgen6] ENABLED - AI Control Active");
            } else {
                System.out.println("[Newgen6] DISABLED");
                releaseInputs(client);
                saveModel();
                previousObservation = null;
                previousAction = -1;
            }
        }

        if (!enabled) {
            return;
        }

        Observation frame = observe(client.player);
        if (frame == null) {
            return;
        }

        if (!historyInitialized) {
            for (int i = 0; i < HISTORY_FRAMES; i++) {
                System.arraycopy(frame.values, 0, HISTORY, i * FRAME_INPUTS, FRAME_INPUTS);
            }
            historyInitialized = true;
        }

        if ((client.player.age & 1) != 0) {
            return;
        }
        
        pushHistory(frame.values);
        Observation current = new Observation(HISTORY);

        if (aiEnabled) {
            runAI(client, current);
        }

        if (client.player.age % 1200 == 0) {
            saveModel();
        }
    }

    private static Observation observe(ClientPlayerEntity player) {
        PlayerEntity enemy = findNearestOpponent(player);
        if (enemy == null) return null;

        double rx = enemy.getX() - player.getX();
        double ry = enemy.getY() - player.getY();
        double rz = enemy.getZ() - player.getZ();

        double distance = Math.sqrt(rx * rx + ry * ry + rz * rz);
        var pv = player.getVelocity();
        var ev = enemy.getVelocity();

        float desiredYaw = (float) Math.toDegrees(Math.atan2(-rx, rz));
        float yawError = wrapDegrees(desiredYaw - player.getYaw());
        double horizontal = Math.sqrt(rx * rx + rz * rz);
        float desiredPitch = (float) -Math.toDegrees(Math.atan2(ry, horizontal));
        float pitchError = desiredPitch - player.getPitch();

        double[] x = new double[FRAME_INPUTS];
        x[0] = clamp(distance / 8.0);
        x[1] = clamp(rx / 8.0);
        x[2] = clamp(ry / 4.0);
        x[3] = clamp(rz / 8.0);
        x[4] = clamp((ev.x - pv.x) / 2.0);
        x[5] = clamp((ev.y - pv.y) / 2.0);
        x[6] = clamp((ev.z - pv.z) / 2.0);
        x[7] = clamp(pv.x / 2.0);
        x[8] = clamp(pv.y / 2.0);
        x[9] = clamp(pv.z / 2.0);
        x[10] = clamp(player.getHealth() / 20.0);
        x[11] = clamp(enemy.getHealth() / 20.0);
        x[12] = clamp(yawError / 180.0);
        x[13] = clamp(pitchError / 90.0);
        x[14] = player.getAttackCooldownProgress(0.0f);
        x[15] = player.isSprinting() ? 1.0 : 0.0;
        x[16] = enemy.isSprinting() ? 1.0 : 0.0;
        x[17] = player.isOnGround() ? 1.0 : 0.0;
        x[18] = enemy.isOnGround() ? 1.0 : 0.0;

        return new Observation(x);
    }

    private static PlayerEntity findNearestOpponent(ClientPlayerEntity player) {
        PlayerEntity best = null;
        double bestDistanceSq = 12.0 * 12.0;

        for (PlayerEntity other : player.getEntityWorld().getPlayers()) {
            if (other == player || other.isDead()) continue;
            double d = player.squaredDistanceTo(other);
            if (d < bestDistanceSq) {
                bestDistanceSq = d;
                best = other;
            }
        }
        return best;
    }

    private static void runAI(MinecraftClient client, Observation observation) {
        ClientPlayerEntity player = client.player;
        if (player == null) return;

        double[] state = observation.values.clone();
        int action = POLICY.chooseAction(state, 1.0);

        if (action < ACTION_REGISTRY.size()) {
            ACTION_REGISTRY.get(action).executor.accept(client);
        } else {
            ACTION_REGISTRY.get(0).executor.accept(client);
        }

        Observation frame = observe(client.player);
        if (frame == null) return;
        
        pushHistory(frame.values);
        Observation nextState = new Observation(HISTORY);

        double reward = calculateUnifiedReward(previousObservation, nextState, action);

        if (previousObservation != null) {
            accumulatedReturn += gammaMultiplier * reward;
            gammaMultiplier *= GAMMA;

            POLICY.train(previousObservation.values, previousAction, accumulatedReturn, LEARNING_RATE);
            
            if (player.age % 20 == 0) {
                System.out.println("[Newgen6] Action ID: " + action + 
                                   " | Return: " + String.format(java.util.Locale.ROOT, "%.4f", accumulatedReturn));
            }
        }

        previousObservation = nextState;
        previousAction = action;
    }

    private static double calculateUnifiedReward(Observation oldState, Observation newState, int action) {
        if (oldState == null) return 0.0;
        final int last = INPUTS - FRAME_INPUTS;

        double oldPlayerHP = oldState.values[last + 10];
        double newPlayerHP = newState.values[last + 10];
        double oldEnemyHP = oldState.values[last + 11];
        double newEnemyHP = newState.values[last + 11];

        double damageDealt = Math.max(0.0, oldEnemyHP - newEnemyHP);
        double damageTaken = Math.max(0.0, oldPlayerHP - newPlayerHP);

        double reward = (damageDealt * 30.0) - (damageTaken * 35.0);

        double beforeYaw = oldState.values[last + 12] * 180.0;
        double beforePitch = oldState.values[last + 13] * 90.0;
        double afterYaw = newState.values[last + 12] * 180.0;
        double afterPitch = newState.values[last + 13] * 90.0;

        double beforeError = Math.hypot(beforeYaw, beforePitch);
        double afterError = Math.hypot(afterYaw, afterPitch);

        reward += (beforeError - afterError) / 15.0;

        if (afterError < 4.0) {
            reward += 0.3;
        }

        if (action == ACTION_REGISTRY.size() - 1 && oldState.values[last + 14] < 0.9f) {
            reward -= 0.2;
        }

        return Math.max(-50.0, Math.min(50.0, reward));
    }
    private static float clampPitch(float pitch) { 
        return Math.max(-90.0f, Math.min(90.0f, pitch)); 
    }

    private static void pushHistory(double[] frame) {
        System.arraycopy(HISTORY, FRAME_INPUTS, HISTORY, 0, INPUTS - FRAME_INPUTS);
        System.arraycopy(frame, 0, HISTORY, INPUTS - FRAME_INPUTS, FRAME_INPUTS);
    }

    private static void releaseInputs(MinecraftClient client) {
        client.options.attackKey.setPressed(false);
    }

    private record AIAction(String name, Consumer<MinecraftClient> executor) {}

    private static final class NeuralPolicy {
        private final int input;
        private final int hidden1;
        private final int hidden2;
        private final int output;

        private final double[][] w1, w2, w3;
        private final double[] b1, b2, b3;

        private final double[][] mw1, mw2, mw3;
        private final double[][] vw1, vw2, vw3;
        private final double[] mb1, mb2, mb3;
        private final double[] vb1, vb2, vb3;
        private int adamStep = 0;

        private double averageReturn = 0.0;

        NeuralPolicy(int input, int hidden1, int hidden2, int output) {
            this.input = input; this.hidden1 = hidden1; this.hidden2 = hidden2; this.output = output;
            w1 = new double[input][hidden1]; w2 = new double[hidden1][hidden2]; w3 = new double[hidden2][output];
            b1 = new double[hidden1]; b2 = new double[hidden2]; b3 = new double[output];

            mw1 = new double[input][hidden1]; mw2 = new double[hidden1][hidden2]; mw3 = new double[hidden2][output];
            vw1 = new double[input][hidden1]; vw2 = new double[hidden1][hidden2]; vw3 = new double[hidden2][output];
            mb1 = new double[hidden1]; mb2 = new double[hidden2]; mb3 = new double[output];
            vb1 = new double[hidden1]; vb2 = new double[hidden2]; vb3 = new double[output];

            initialize();
        }

        private void initialize() {
            randomize(w1, input); randomize(w2, hidden1); randomize(w3, hidden2);
        }

        private void randomize(double[][] matrix, int fanIn) {
            double scale = Math.sqrt(2.0 / fanIn);
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    matrix[i][j] = RANDOM.nextGaussian() * scale;
                }
            }
        }

        private Forward forward(double[] x) {
            double[] h1 = new double[hidden1];
            double[] h2 = new double[hidden2];
            double[] logits = new double[output];

            for (int j = 0; j < hidden1; j++) {
                double sum = b1[j];
                for (int i = 0; i < input; i++) sum += x[i] * w1[i][j];
                h1[j] = Math.max(0.0, sum);
            }
            for (int j = 0; j < hidden2; j++) {
                double sum = b2[j];
                for (int i = 0; i < hidden1; i++) sum += h1[i] * w2[i][j];
                h2[j] = Math.max(0.0, sum);
            }
            for (int j = 0; j < output; j++) {
                double sum = b3[j];
                for (int i = 0; i < hidden2; i++) sum += h2[i] * w3[i][j];
                logits[j] = sum;
            }
            return new Forward(h1, h2, logits);
        }

        private int chooseAction(double[] x, double temperature) {
            Forward f = forward(x);
            double[] probabilities = softmax(f.logits, temperature);
            double r = RANDOM.nextDouble();
            double cumulative = 0.0;
            for (int i = 0; i < probabilities.length; i++) {
                cumulative += probabilities[i];
                if (r <= cumulative) return i;
            }
            return probabilities.length - 1;
        }

        private void train(double[] x, int action, double returnVal, double lr) {
            adamStep++;
            Forward f = forward(x);
            double[] probabilities = softmax(f.logits, 1.0);
            double[] dLogits = new double[output];

            averageReturn = (averageReturn * 0.99) + (returnVal * 0.01);
            double advantage = returnVal - averageReturn;

            for (int i = 0; i < output; i++) {
                double entropyGrad = ENTROPY_BETA * probabilities[i] * (Math.log(probabilities[i] + 1e-8));
                dLogits[i] = (-probabilities[i] * advantage) + entropyGrad;
            }
            dLogits[action] += advantage;

            double[][] dw3 = new double[hidden2][output];
            double[] db3 = new double[output];
            for (int i = 0; i < hidden2; i++) {
                for (int j = 0; j < output; j++) {
                    dw3[i][j] = dLogits[j] * f.h2[i];
                }
            }
            System.arraycopy(dLogits, 0, db3, 0, output);
            updateAdamMatrix(w3, mw3, vw3, dw3, lr);
            updateAdamArray(b3, mb3, vb3, db3, lr);

            double[] dh2 = new double[hidden2];
            for (int i = 0; i < hidden2; i++) {
                double sum = 0.0;
                for (int j = 0; j < output; j++) sum += dLogits[j] * w3[i][j];
                dh2[i] = (f.h2[i] > 0.0) ? sum : 0.0;
            }

            double[][] dw2 = new double[hidden1][hidden2];
            double[] db2 = new double[hidden2];
            for (int i = 0; i < hidden1; i++) {
                for (int j = 0; j < hidden2; j++) {
                    dw2[i][j] = dh2[j] * f.h1[i];
                }
            }
            System.arraycopy(dh2, 0, db2, 0, hidden2);
            updateAdamMatrix(w2, mw2, vw2, dw2, lr);
            updateAdamArray(b2, mb2, vb2, db2, lr);

            double[] dh1 = new double[hidden1];
            for (int i = 0; i < hidden1; i++) {
                double sum = 0.0;
                for (int j = 0; j < hidden2; j++) sum += dh2[j] * w2[i][j];
                dh1[i] = (f.h1[i] > 0.0) ? sum : 0.0;
            }

            double[][] dw1 = new double[input][hidden1];
            double[] db1 = new double[hidden1];
            for (int i = 0; i < input; i++) {
                for (int j = 0; j < hidden1; j++) {
                    dw1[i][j] = dh1[j] * x[i];
                }
            }
            System.arraycopy(dh1, 0, db1, 0, hidden1);
            updateAdamMatrix(w1, mw1, vw1, dw1, lr);
            updateAdamArray(b1, mb1, vb1, db1, lr);
        }

        private void updateAdamMatrix(double[][] w, double[][] m, double[][] v, double[][] grad, double lr) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[i].length; j++) {
                    m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * grad[i][j];
                    v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * grad[i][j] * grad[i][j];
                    double mHat = m[i][j] / (1.0 - Math.pow(beta1, adamStep));
                    double vHat = v[i][j] / (1.0 - Math.pow(beta2, adamStep));
                    w[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
                }
            }
        }

        private void updateAdamArray(double[] w, double[] m, double[] v, double[] grad, double lr) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            for (int i = 0; i < w.length; i++) {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                double mHat = m[i] / (1.0 - Math.pow(beta1, adamStep));
                double vHat = v[i] / (1.0 - Math.pow(beta2, adamStep));
                w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
        }

        private double[] softmax(double[] logits, double temperature) {
            double[] probabilities = new double[logits.length];
            double max = Arrays.stream(logits).max().orElse(0.0);
            double sum = 0.0;
            for (int i = 0; i < logits.length; i++) {
                probabilities[i] = Math.exp((logits[i] - max) / temperature);
                sum += probabilities[i];
            }
            for (int i = 0; i < probabilities.length; i++) probabilities[i] /= sum;
            return probabilities;
        }
    }

    private record Forward(double[] h1, double[] h2, double[] logits) {}

    private static final class Observation implements Serializable {
        private final double[] values;
        private Observation(double[] values) { this.values = Arrays.copyOf(values, values.length); }
    }

    private static void saveModel() {
        if (MODEL_FILE == null) return;
        try {
            Files.createDirectories(DATA_DIR);
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(MODEL_FILE)))) {
                out.writeInt(1);
                writeMatrix(out, POLICY.w1); writeMatrix(out, POLICY.w2); writeMatrix(out, POLICY.w3);
                writeArray(out, POLICY.b1); writeArray(out, POLICY.b2); writeArray(out, POLICY.b3);
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    private static void loadModel() {
        if (MODEL_FILE == null || !Files.exists(MODEL_FILE)) return;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(MODEL_FILE)))) {
            if (in.readInt() != 1) throw new IOException("Unsupported model version.");
            readMatrix(in, POLICY.w1); readMatrix(in, POLICY.w2); readMatrix(in, POLICY.w3);
            readArray(in, POLICY.b1); readArray(in, POLICY.b2); readArray(in, POLICY.b3);
        } catch (IOException e) { e.printStackTrace(); }
    }

    private static void writeMatrix(DataOutputStream out, double[][] matrix) throws IOException {
        out.writeInt(matrix.length);
        for (double[] row : matrix) { out.writeInt(row.length); for (double value : row) out.writeDouble(value); }
    }

    private static void readMatrix(DataInputStream in, double[][] matrix) throws IOException {
        int rows = in.readInt();
        for (int i = 0; i < rows; i++) {
            int cols = in.readInt();
            for (int j = 0; j < cols; j++) matrix[i][j] = in.readDouble();
        }
    }

    private static void writeArray(DataOutputStream out, double[] array) throws IOException {
        out.writeInt(array.length);
        for (double value : array) out.writeDouble(value);
    }

    private static void readArray(DataInputStream in, double[] array) throws IOException {
        int length = in.readInt();
        for (int i = 0; i < length; i++) array[i] = in.readDouble();
    }

    private static double clamp(double value) { return Math.max(-1.0, Math.min(1.0, value)); }
    private static float wrapDegrees(float degrees) {
        while (degrees >= 180.0f) degrees -= 360.0f;
        while (degrees < -180.0f) degrees += 360.0f;
        return degrees;
    }
}
