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
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;

import org.lwjgl.glfw.GLFW;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Newgen6 - Fixed strictly On-Policy REINFORCE Agent.
 *
 * C = master toggle.
 */
public final class Newgen6Client implements ClientModInitializer {

    private static final String MOD_ID = "newgen6";

    private static final KeyBinding.Category NEWGEN6_CATEGORY =
            KeyBinding.Category.create(
                    Identifier.of("newgen6", "general")
            );

    private static final boolean ENABLE_AI_CONTROL_AFTER_TRAINING = true;

    private static Path DATA_DIR;
    private static Path MODEL_FILE;
    private static KeyBinding MASTER_KEY;

    private static boolean enabled = false;
    private static boolean training = false;
    private static boolean aiEnabled = false;

    private static final int FRAME_INPUTS = 19;
    private static final int HISTORY_FRAMES = 50; // 5 seconds at 10 Hz
    private static final int INPUTS = FRAME_INPUTS * HISTORY_FRAMES;
    private static final int ACTIONS = 20;

    private static final float[] AIM_YAW = {-4f, 4f, 0f, 0f, -3f, 3f};
    private static final float[] AIM_PITCH = {0f, 0f, -3f, 3f, -2f, -2f};

    private static final Random RANDOM = new Random();

    private static final NeuralPolicy POLICY =
            new NeuralPolicy(INPUTS, 64, 64, ACTIONS);

    private static final double LEARNING_RATE = 0.00015;

    private static final double[] HISTORY = new double[INPUTS];
    private static int historyTick;

    private static Observation previousObservation;
    private static int previousAction = -1;

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

        loadModel();

        MASTER_KEY = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.master",
                        GLFW.GLFW_KEY_C,
                        NEWGEN6_CATEGORY
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(Newgen6Client::tick);

        System.out.println("[Newgen6] Loaded (Strict On-Policy Mode).");
    }

    private static void tick(MinecraftClient client) {
        if (client.player == null || client.world == null) {
            return;
        }

        while (MASTER_KEY.wasPressed()) {
            enabled = !enabled;
            training = enabled;
            aiEnabled = enabled; // Forces AI control when enabled

            if (enabled) {
                previousObservation = null;
                previousAction = -1;
                Arrays.fill(HISTORY, 0.0);
                historyTick = 0;

                System.out.println("[Newgen6] ENABLED - AI taking control");
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

        // Let the Softmax distribution handle exploration naturally.
        int action = POLICY.chooseAction(state, 1.0);

        applyAction(client, action);
        if (action >= 14) {
            applyAimAction(client, action - 14);
        }

        Observation frame = observe(client.player);
        if (frame == null) return;
        
        // Push the new result to history to create the "next state"
        pushHistory(frame.values);
        Observation nextState = new Observation(HISTORY);

        if (previousObservation != null) {
            // Compare previous full history to new full history
            double reward = calculateUnifiedReward(previousObservation, nextState);
            POLICY.train(previousObservation.values, previousAction, reward, LEARNING_RATE);
            
            if (player.age % 20 == 0) {
                System.out.println("[Newgen6] AI Action: " + action + 
                                   " | Reward: " + String.format(java.util.Locale.ROOT, "%.4f", reward));
            }
        }

        previousObservation = nextState;
        previousAction = action;
    }

    private static double calculateUnifiedReward(Observation oldState, Observation newState) {
        final int last = INPUTS - FRAME_INPUTS;

        double oldPlayerHP = oldState.values[last + 10];
        double newPlayerHP = newState.values[last + 10];
        double oldEnemyHP = oldState.values[last + 11];
        double newEnemyHP = newState.values[last + 11];

        double damageDealt = oldEnemyHP - newEnemyHP;
        double damageTaken = oldPlayerHP - newPlayerHP;

        double reward = (damageDealt * 20.0) - (damageTaken * 25.0);

        // Distance reward
        double distance = newState.values[last] * 8.0;
        if (distance >= 2.0 && distance <= 4.0) reward += 0.02;

        // Aim reward (indexing the most recent frame in history)
        double beforeYaw = oldState.values[last + 12] * 180.0;
        double beforePitch = oldState.values[last + 13] * 90.0;
        double afterYaw = newState.values[last + 12] * 180.0;
        double afterPitch = newState.values[last + 13] * 90.0;

        double beforeError = Math.hypot(beforeYaw, beforePitch);
        double afterError = Math.hypot(afterYaw, afterPitch);

        reward += (beforeError - afterError) / 30.0;
        if (afterError < 15.0) reward += 0.02;

        if (newPlayerHP <= 0.001) reward -= 100.0;
        if (newEnemyHP <= 0.001) reward += 100.0;

        // Small living/time penalty to force action
        reward -= 0.001;
        return Math.max(-100.0, Math.min(100.0, reward));
    }

    private static void applyAction(MinecraftClient client, int action) {
        releaseMovement(client);
        switch (action) {
            case 1 -> client.options.forwardKey.setPressed(true);
            case 2 -> client.options.backKey.setPressed(true);
            case 3 -> client.options.leftKey.setPressed(true);
            case 4 -> client.options.rightKey.setPressed(true);
            case 5 -> { client.options.forwardKey.setPressed(true); client.options.leftKey.setPressed(true); }
            case 6 -> { client.options.forwardKey.setPressed(true); client.options.rightKey.setPressed(true); }
            case 7 -> { client.options.backKey.setPressed(true); client.options.leftKey.setPressed(true); }
            case 8 -> { client.options.backKey.setPressed(true); client.options.rightKey.setPressed(true); }
            case 9 -> attack(client);
            case 10 -> { client.options.forwardKey.setPressed(true); attack(client); }
            case 11 -> { client.options.leftKey.setPressed(true); attack(client); }
            case 12 -> { client.options.rightKey.setPressed(true); attack(client); }
            case 13 -> client.options.jumpKey.setPressed(true);
        }
    }

    private static void applyAimAction(MinecraftClient client, int aimAction) {
        if (client.player == null || aimAction < 0 || aimAction >= AIM_YAW.length) return;
        client.player.setYaw(client.player.getYaw() + AIM_YAW[aimAction]);
        client.player.setPitch(clampPitch(client.player.getPitch() + AIM_PITCH[aimAction]));
    }

    private static float clampPitch(float pitch) { return Math.max(-90.0f, Math.min(90.0f, pitch)); }

    private static void attack(MinecraftClient client) {
        if (client.interactionManager == null || client.player == null) return;
        if (client.targetedEntity instanceof PlayerEntity target) {
            client.interactionManager.attackEntity(client.player, target);
            client.player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
        }
    }

    private static void pushHistory(double[] frame) {
        System.arraycopy(HISTORY, FRAME_INPUTS, HISTORY, 0, INPUTS - FRAME_INPUTS);
        System.arraycopy(frame, 0, HISTORY, INPUTS - FRAME_INPUTS, FRAME_INPUTS);
        historyTick++;
    }

    private static void releaseMovement(MinecraftClient client) {
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
    }

    private static void releaseInputs(MinecraftClient client) {
        releaseMovement(client);
        client.options.attackKey.setPressed(false);
    }

    private static final class NeuralPolicy {
        private final int input;
        private final int hidden1;
        private final int hidden2;
        private final int output;

        private final double[][] w1, w2, w3;
        private final double[] b1, b2, b3;

        // Advantage Baseline 
        private double averageReward = 0.0;

        NeuralPolicy(int input, int hidden1, int hidden2, int output) {
            this.input = input; this.hidden1 = hidden1; this.hidden2 = hidden2; this.output = output;
            w1 = new double[input][hidden1]; w2 = new double[hidden1][hidden2]; w3 = new double[hidden2][output];
            b1 = new double[hidden1]; b2 = new double[hidden2]; b3 = new double[output];
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

        private void train(double[] x, int action, double reward, double lr) {
            Forward f = forward(x);
            double[] probabilities = softmax(f.logits, 1.0);
            double[] dLogits = new double[output];

            // Update moving baseline and calculate advantage
            averageReward = (averageReward * 0.99) + (reward * 0.01);
            double advantage = reward - averageReward;

            for (int i = 0; i < output; i++) {
                dLogits[i] = -probabilities[i] * advantage;
            }
            dLogits[action] += advantage;

            for (int i = 0; i < hidden2; i++) {
                for (int j = 0; j < output; j++) {
                    w3[i][j] += lr * dLogits[j] * f.h2[i];
                }
            }
            for (int j = 0; j < output; j++) b3[j] += lr * dLogits[j];

            double[] dh2 = new double[hidden2];
            for (int i = 0; i < hidden2; i++) {
                double sum = 0.0;
                for (int j = 0; j < output; j++) sum += dLogits[j] * w3[i][j];
                dh2[i] = (f.h2[i] > 0.0) ? sum : 0.0;
            }

            double[] dh1 = new double[hidden1];
            for (int i = 0; i < hidden1; i++) {
                double sum = 0.0;
                for (int j = 0; j < hidden2; j++) sum += dh2[j] * w2[i][j];
                dh1[i] = (f.h1[i] > 0.0) ? sum : 0.0;
            }

            for (int i = 0; i < hidden1; i++) {
                for (int j = 0; j < hidden2; j++) w2[i][j] += lr * dh2[j] * f.h1[i];
            }
            for (int j = 0; j < hidden2; j++) b2[j] += lr * dh2[j];

            for (int i = 0; i < input; i++) {
                for (int j = 0; j < hidden1; j++) w1[i][j] += lr * dh1[j] * x[i];
            }
            for (int j = 0; j < hidden1; j++) b1[j] += lr * dh1[j];
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