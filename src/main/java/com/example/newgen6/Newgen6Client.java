package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;

import net.minecraft.client.MinecraftClient;
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
 * Newgen6 - self-contained sword PvP data collector + lightweight
 * policy-gradient trainer.
 *
 * C = master toggle.
 *
 * When enabled:
 *   - records human sword PvP experience
 *   - trains the neural policy online
 *
 * AI control is deliberately OFF while collecting human demonstrations.
 * Set ENABLE_AI_CONTROL_AFTER_TRAINING to true if you want C to hand
 * control to the learned policy immediately.
 */
public class Newgen6Client implements ClientModInitializer {

    private static final String MOD_ID = "newgen6";

    /*
     * Change this to true only when you want C to enable AI control too.
     * Default false = C records + trains from your gameplay.
     */
    private static final boolean ENABLE_AI_CONTROL_AFTER_TRAINING = false;

    private static Path DATA_DIR;
    private static Path MODEL_FILE;
    private static Path EXPERIENCE_FILE;

    private static KeyBinding MASTER_KEY;

    private static boolean enabled = false;
    private static boolean recording = false;
    private static boolean training = false;
    private static boolean aiEnabled = false;

    /*
     * Observation:
     * 0  distance
     * 1  relative X
     * 2  relative Y
     * 3  relative Z
     * 4  relative velocity X
     * 5  relative velocity Y
     * 6  relative velocity Z
     * 7  player velocity X
     * 8  player velocity Y
     * 9  player velocity Z
     * 10 player health
     * 11 enemy health
     * 12 yaw error
     * 13 pitch error
     * 14 attack cooldown
     * 15 player sprinting
     * 16 enemy sprinting
     * 17 player on ground
     * 18 enemy on ground
     */
    private static final int INPUTS = 19;

    /*
     * Actions:
     * 0  idle
     * 1  forward
     * 2  backward
     * 3  left
     * 4  right
     * 5  forward-left
     * 6  forward-right
     * 7  backward-left
     * 8  backward-right
     * 9  attack
     * 10 forward + attack
     * 11 left + attack
     * 12 right + attack
     * 13 jump
     */
    private static final int ACTIONS = 14;

    private static final NeuralPolicy POLICY =
            new NeuralPolicy(INPUTS, 64, 64, ACTIONS);

    private static final Random RANDOM = new Random();

    private static final double LEARNING_RATE = 0.00015;
    private static final int MAX_EXPERIENCES = 250_000;

    private static final ArrayDeque<Experience> EXPERIENCE =
            new ArrayDeque<>();

    private static Observation previousObservation;
    private static int previousAction = -1;

    @Override
    public void onInitializeClient() {
        MinecraftClient client = MinecraftClient.getInstance();

        DATA_DIR = client.runDirectory.toPath().resolve("newgen6");
        MODEL_FILE = DATA_DIR.resolve("model.bin");
        EXPERIENCE_FILE = DATA_DIR.resolve("experience.bin");

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

        ClientTickEvents.END_CLIENT_TICK.register(
                Newgen6Client::tick
        );

        System.out.println("[Newgen6] Loaded.");
        System.out.println("[Newgen6] C = enable/disable");
        System.out.println("[Newgen6] Data directory: " + DATA_DIR);
    }

    private static void tick(MinecraftClient client) {
        if (client.player == null || client.world == null) {
            return;
        }

        while (MASTER_KEY.wasPressed()) {
            enabled = !enabled;

            recording = enabled;
            training = enabled;
            aiEnabled =
                    enabled && ENABLE_AI_CONTROL_AFTER_TRAINING;

            if (enabled) {
                System.out.println("[Newgen6] ENABLED");
                System.out.println(
                        "[Newgen6] Recording: " + recording +
                        " | Training: " + training +
                        " | AI: " + aiEnabled
                );
            } else {
                System.out.println("[Newgen6] DISABLED");
                releaseInputs(client);
                saveModel();
                saveExperience();
                previousObservation = null;
                previousAction = -1;
            }
        }

        if (!enabled) {
            return;
        }

        Observation current = observe(client.player);

        if (current == null) {
            return;
        }

        /*
         * Human demonstration collection.
         */
        if (recording && !aiEnabled) {
            int action = getHumanAction(client);

            if (previousObservation != null &&
                    previousAction >= 0) {

                double reward =
                        calculateReward(
                                previousObservation,
                                current
                        );

                addExperience(
                        new Experience(
                                previousObservation.values,
                                previousAction,
                                reward,
                                current.values
                        )
                );

                /*
                 * Online learning.
                 */
                if (training && client.player.age % 2 == 0) {
                    trainOne();
                }
            }

            previousObservation = current;
            previousAction = action;
        }

        /*
         * Optional autonomous AI mode.
         */
        if (aiEnabled) {
            runAI(client, current);

            if (training && client.player.age % 2 == 0) {
                trainOne();
            }
        }

        /*
         * Save every 60 seconds.
         */
        if (client.player.age % 1200 == 0) {
            saveModel();
            saveExperience();
        }
    }

    private static Observation observe(ClientPlayerEntity player) {
        PlayerEntity enemy = findNearestOpponent(player);

        if (enemy == null) {
            return null;
        }

        double rx = enemy.getX() - player.getX();
        double ry = enemy.getY() - player.getY();
        double rz = enemy.getZ() - player.getZ();

        double distance =
                Math.sqrt(rx * rx + ry * ry + rz * rz);

        var pv = player.getVelocity();
        var ev = enemy.getVelocity();

        float desiredYaw =
                (float) Math.toDegrees(
                        Math.atan2(-rx, rz)
                );

        float yawError =
                wrapDegrees(desiredYaw - player.getYaw());

        double horizontal =
                Math.sqrt(rx * rx + rz * rz);

        float desiredPitch =
                (float) -Math.toDegrees(
                        Math.atan2(ry, horizontal)
                );

        float pitchError =
                desiredPitch - player.getPitch();

        float cooldown =
                player.getAttackCooldownProgress(0.0f);

        double[] x = new double[INPUTS];

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

        x[14] = cooldown;

        x[15] = player.isSprinting() ? 1.0 : 0.0;
        x[16] = enemy.isSprinting() ? 1.0 : 0.0;
        x[17] = player.isOnGround() ? 1.0 : 0.0;
        x[18] = enemy.isOnGround() ? 1.0 : 0.0;

        return new Observation(x);
    }

    private static PlayerEntity findNearestOpponent(
            ClientPlayerEntity player
    ) {
        PlayerEntity best = null;
        double bestDistanceSq = 12.0 * 12.0;

        for (PlayerEntity other :
                player.getEntityWorld().getPlayers()) {

            if (other == player || other.isDead()) {
                continue;
            }

            double d = player.squaredDistanceTo(other);

            if (d < bestDistanceSq) {
                bestDistanceSq = d;
                best = other;
            }
        }

        return best;
    }

    private static int getHumanAction(MinecraftClient client) {
        if (client.options.attackKey.isPressed()) {
            if (client.options.leftKey.isPressed()) return 11;
            if (client.options.rightKey.isPressed()) return 12;
            if (client.options.forwardKey.isPressed()) return 10;
            return 9;
        }

        boolean forward =
                client.options.forwardKey.isPressed();
        boolean backward =
                client.options.backKey.isPressed();
        boolean left =
                client.options.leftKey.isPressed();
        boolean right =
                client.options.rightKey.isPressed();
        boolean jump =
                client.options.jumpKey.isPressed();

        if (jump) return 13;
        if (forward && left) return 5;
        if (forward && right) return 6;
        if (backward && left) return 7;
        if (backward && right) return 8;
        if (forward) return 1;
        if (backward) return 2;
        if (left) return 3;
        if (right) return 4;

        return 0;
    }

    private static double calculateReward(
            Observation oldState,
            Observation newState
    ) {
        double oldPlayerHP = oldState.values[10];
        double newPlayerHP = newState.values[10];

        double oldEnemyHP = oldState.values[11];
        double newEnemyHP = newState.values[11];

        double damageDealt = oldEnemyHP - newEnemyHP;
        double damageTaken = oldPlayerHP - newPlayerHP;

        double reward = 0.0;

        reward += damageDealt * 20.0;
        reward -= damageTaken * 25.0;

        double distance = newState.values[0] * 8.0;

        /*
         * Small spacing reward. This is deliberately weak so the
         * AI does not simply camp at a fixed distance.
         */
        if (distance >= 2.0 && distance <= 4.0) {
            reward += 0.02;
        }

        double yawError =
                Math.abs(newState.values[12] * 180.0);

        if (yawError < 15.0) {
            reward += 0.02;
        }

        if (newPlayerHP <= 0.001) {
            reward -= 100.0;
        }

        if (newEnemyHP <= 0.001) {
            reward += 100.0;
        }

        reward -= 0.001;

        return reward;
    }

    private static void addExperience(Experience e) {
        if (EXPERIENCE.size() >= MAX_EXPERIENCES) {
            EXPERIENCE.pollFirst();
        }

        EXPERIENCE.addLast(e);
    }

    private static void trainOne() {
        if (EXPERIENCE.isEmpty()) return;

        int index = RANDOM.nextInt(EXPERIENCE.size());

        Iterator<Experience> iterator =
                EXPERIENCE.iterator();

        for (int i = 0; i < index; i++) {
            iterator.next();
        }

        Experience e = iterator.next();

        POLICY.train(
                e.state,
                e.action,
                e.reward,
                LEARNING_RATE
        );
    }

    private static void runAI(
            MinecraftClient client,
            Observation observation
    ) {
        ClientPlayerEntity player = client.player;

        if (player == null || !holdingSword(player)) {
            releaseInputs(client);
            return;
        }

        int action =
                POLICY.chooseAction(
                        observation.values,
                        1.0
                );

        applyAction(client, action);
    }

    private static boolean holdingSword(
            ClientPlayerEntity player
    ) {
        return isSword(player.getMainHandStack()) ||
                isSword(player.getOffHandStack());
    }

    private static boolean isSword(ItemStack stack) {
        return stack.isOf(Items.WOODEN_SWORD) ||
                stack.isOf(Items.STONE_SWORD) ||
                stack.isOf(Items.IRON_SWORD) ||
                stack.isOf(Items.GOLDEN_SWORD) ||
                stack.isOf(Items.DIAMOND_SWORD) ||
                stack.isOf(Items.NETHERITE_SWORD);
    }

    private static void applyAction(
            MinecraftClient client,
            int action
    ) {
        releaseMovement(client);

        switch (action) {
            case 1 ->
                    client.options.forwardKey.setPressed(true);

            case 2 ->
                    client.options.backKey.setPressed(true);

            case 3 ->
                    client.options.leftKey.setPressed(true);

            case 4 ->
                    client.options.rightKey.setPressed(true);

            case 5 -> {
                client.options.forwardKey.setPressed(true);
                client.options.leftKey.setPressed(true);
            }

            case 6 -> {
                client.options.forwardKey.setPressed(true);
                client.options.rightKey.setPressed(true);
            }

            case 7 -> {
                client.options.backKey.setPressed(true);
                client.options.leftKey.setPressed(true);
            }

            case 8 -> {
                client.options.backKey.setPressed(true);
                client.options.rightKey.setPressed(true);
            }

            case 9 ->
                    attack(client);

            case 10 -> {
                client.options.forwardKey.setPressed(true);
                attack(client);
            }

            case 11 -> {
                client.options.leftKey.setPressed(true);
                attack(client);
            }

            case 12 -> {
                client.options.rightKey.setPressed(true);
                attack(client);
            }

            case 13 ->
                    client.options.jumpKey.setPressed(true);

            default -> {
            }
        }
    }

    private static void attack(MinecraftClient client) {
        if (client.interactionManager == null ||
                client.player == null) {
            return;
        }

        if (client.targetedEntity instanceof PlayerEntity target) {
            client.interactionManager.attackEntity(
                    client.player,
                    target
            );

            client.player.swingHand(
                    net.minecraft.util.Hand.MAIN_HAND
            );
        }
    }

    private static void releaseMovement(
            MinecraftClient client
    ) {
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
    }

    private static void releaseInputs(
            MinecraftClient client
    ) {
        releaseMovement(client);
        client.options.attackKey.setPressed(false);
    }

    /* ============================================================
       NEURAL POLICY
       ============================================================ */

    private static final class NeuralPolicy {
        private final int input;
        private final int hidden1;
        private final int hidden2;
        private final int output;

        private final double[][] w1;
        private final double[][] w2;
        private final double[][] w3;

        private final double[] b1;
        private final double[] b2;
        private final double[] b3;

        NeuralPolicy(
                int input,
                int hidden1,
                int hidden2,
                int output
        ) {
            this.input = input;
            this.hidden1 = hidden1;
            this.hidden2 = hidden2;
            this.output = output;

            w1 = new double[input][hidden1];
            w2 = new double[hidden1][hidden2];
            w3 = new double[hidden2][output];

            b1 = new double[hidden1];
            b2 = new double[hidden2];
            b3 = new double[output];

            initialize();
        }

        private void initialize() {
            randomize(w1, input);
            randomize(w2, hidden1);
            randomize(w3, hidden2);
        }

        private void randomize(
                double[][] matrix,
                int fanIn
        ) {
            double scale =
                    Math.sqrt(2.0 / fanIn);

            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0;
                     j < matrix[i].length;
                     j++) {
                    matrix[i][j] =
                            RANDOM.nextGaussian() * scale;
                }
            }
        }

        private Forward forward(double[] x) {
            double[] h1 = new double[hidden1];
            double[] h2 = new double[hidden2];
            double[] logits = new double[output];

            for (int j = 0; j < hidden1; j++) {
                double sum = b1[j];

                for (int i = 0; i < input; i++) {
                    sum += x[i] * w1[i][j];
                }

                h1[j] = Math.max(0.0, sum);
            }

            for (int j = 0; j < hidden2; j++) {
                double sum = b2[j];

                for (int i = 0; i < hidden1; i++) {
                    sum += h1[i] * w2[i][j];
                }

                h2[j] = Math.max(0.0, sum);
            }

            for (int j = 0; j < output; j++) {
                double sum = b3[j];

                for (int i = 0; i < hidden2; i++) {
                    sum += h2[i] * w3[i][j];
                }

                logits[j] = sum;
            }

            return new Forward(h1, h2, logits);
        }

        private int chooseAction(
                double[] x,
                double temperature
        ) {
            Forward f = forward(x);
            double[] probabilities =
                    softmax(f.logits, temperature);

            double r = RANDOM.nextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < probabilities.length; i++) {
                cumulative += probabilities[i];

                if (r <= cumulative) {
                    return i;
                }
            }

            return probabilities.length - 1;
        }

        private void train(
                double[] x,
                int action,
                double reward,
                double lr
        ) {
            /*
             * Keep the first implementation computationally light
             * enough for Pojav/Android.
             */
            reward = Math.max(-100.0,
                    Math.min(100.0, reward));

            Forward f = forward(x);

            double[] probabilities =
                    softmax(f.logits, 1.0);

            double[] dLogits =
                    new double[output];

            /*
             * REINFORCE:
             * gradient log pi(a|s) * reward
             */
            for (int i = 0; i < output; i++) {
                dLogits[i] =
                        -probabilities[i] * reward;
            }

            dLogits[action] += reward;

            for (int i = 0; i < hidden2; i++) {
                for (int j = 0; j < output; j++) {
                    w3[i][j] +=
                            lr * dLogits[j] * f.h2[i];
                }
            }

            for (int j = 0; j < output; j++) {
                b3[j] += lr * dLogits[j];
            }

            double[] dh2 =
                    new double[hidden2];

            for (int i = 0; i < hidden2; i++) {
                double sum = 0.0;

                for (int j = 0; j < output; j++) {
                    sum += dLogits[j] * w3[i][j];
                }

                if (f.h2[i] <= 0.0) {
                    sum = 0.0;
                }

                dh2[i] = sum;
            }

            double[] dh1 =
                    new double[hidden1];

            for (int i = 0; i < hidden1; i++) {
                double sum = 0.0;

                for (int j = 0; j < hidden2; j++) {
                    sum += dh2[j] * w2[i][j];
                }

                if (f.h1[i] <= 0.0) {
                    sum = 0.0;
                }

                dh1[i] = sum;
            }

            for (int i = 0; i < hidden1; i++) {
                for (int j = 0; j < hidden2; j++) {
                    w2[i][j] +=
                            lr * dh2[j] * f.h1[i];
                }
            }

            for (int j = 0; j < hidden2; j++) {
                b2[j] += lr * dh2[j];
            }

            for (int i = 0; i < input; i++) {
                for (int j = 0; j < hidden1; j++) {
                    w1[i][j] +=
                            lr * dh1[j] * x[i];
                }
            }

            for (int j = 0; j < hidden1; j++) {
                b1[j] += lr * dh1[j];
            }

            clipWeights();
        }

        private void clipWeights() {
            clip(w1);
            clip(w2);
            clip(w3);
            clip(b1);
            clip(b2);
            clip(b3);
        }

        private void clip(double[][] matrix) {
            for (double[] row : matrix) {
                for (int i = 0; i < row.length; i++) {
                    row[i] =
                            Math.max(-10.0,
                                    Math.min(10.0, row[i]));
                }
            }
        }

        private void clip(double[] array) {
            for (int i = 0; i < array.length; i++) {
                array[i] =
                        Math.max(-10.0,
                                Math.min(10.0, array[i]));
            }
        }

        private double[] softmax(
                double[] logits,
                double temperature
        ) {
            temperature =
                    Math.max(0.05, temperature);

            double[] probabilities =
                    new double[logits.length];

            double max =
                    Arrays.stream(logits)
                            .max()
                            .orElse(0.0);

            double sum = 0.0;

            for (int i = 0;
                 i < logits.length;
                 i++) {

                probabilities[i] =
                        Math.exp(
                                (logits[i] - max)
                                        / temperature
                        );

                sum += probabilities[i];
            }

            if (sum <= 0.0 ||
                    Double.isNaN(sum)) {

                Arrays.fill(
                        probabilities,
                        1.0 / probabilities.length
                );

                return probabilities;
            }

            for (int i = 0;
                 i < probabilities.length;
                 i++) {

                probabilities[i] /= sum;
            }

            return probabilities;
        }
    }

    private record Forward(
            double[] h1,
            double[] h2,
            double[] logits
    ) {}

    private static final class Observation
            implements Serializable {

        private final double[] values;

        private Observation(double[] values) {
            this.values =
                    Arrays.copyOf(
                            values,
                            values.length
                    );
        }
    }

    private static final class Experience
            implements Serializable {

        private final double[] state;
        private final int action;
        private final double reward;
        private final double[] nextState;

        private Experience(
                double[] state,
                int action,
                double reward,
                double[] nextState
        ) {
            this.state =
                    Arrays.copyOf(state, state.length);

            this.action = action;
            this.reward = reward;

            this.nextState =
                    Arrays.copyOf(
                            nextState,
                            nextState.length
                    );
        }
    }

    /* ============================================================
       MODEL STORAGE
       ============================================================ */

    private static void saveModel() {
        if (MODEL_FILE == null) return;

        try {
            Files.createDirectories(DATA_DIR);

            try (DataOutputStream out =
                         new DataOutputStream(
                                 new BufferedOutputStream(
                                         Files.newOutputStream(
                                                 MODEL_FILE
                                         )
                                 ))) {

                out.writeInt(1);

                writeMatrix(out, POLICY.w1);
                writeMatrix(out, POLICY.w2);
                writeMatrix(out, POLICY.w3);

                writeArray(out, POLICY.b1);
                writeArray(out, POLICY.b2);
                writeArray(out, POLICY.b3);
            }

            System.out.println(
                    "[Newgen6] Model saved."
            );

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void loadModel() {
        if (MODEL_FILE == null ||
                !Files.exists(MODEL_FILE)) {
            return;
        }

        try (DataInputStream in =
                     new DataInputStream(
                             new BufferedInputStream(
                                     Files.newInputStream(
                                             MODEL_FILE
                                     )
                             ))) {

            int version = in.readInt();

            if (version != 1) {
                throw new IOException(
                        "Unsupported model version: "
                                + version
                );
            }

            readMatrix(in, POLICY.w1);
            readMatrix(in, POLICY.w2);
            readMatrix(in, POLICY.w3);

            readArray(in, POLICY.b1);
            readArray(in, POLICY.b2);
            readArray(in, POLICY.b3);

            System.out.println(
                    "[Newgen6] Model loaded."
            );

        } catch (IOException e) {
            System.err.println(
                    "[Newgen6] Could not load model."
            );
            e.printStackTrace();
        }
    }

    private static void saveExperience() {
        if (EXPERIENCE_FILE == null) return;

        try {
            Files.createDirectories(DATA_DIR);

            try (DataOutputStream out =
                         new DataOutputStream(
                                 new BufferedOutputStream(
                                         Files.newOutputStream(
                                                 EXPERIENCE_FILE
                                         )
                                 ))) {

                out.writeInt(EXPERIENCE.size());

                for (Experience e : EXPERIENCE) {
                    writeArray(out, e.state);
                    out.writeInt(e.action);
                    out.writeDouble(e.reward);
                    writeArray(out, e.nextState);
                }
            }

            System.out.println(
                    "[Newgen6] Experiences saved: "
                            + EXPERIENCE.size()
            );

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeMatrix(
            DataOutputStream out,
            double[][] matrix
    ) throws IOException {
        out.writeInt(matrix.length);

        for (double[] row : matrix) {
            out.writeInt(row.length);

            for (double value : row) {
                out.writeDouble(value);
            }
        }
    }

    private static void readMatrix(
            DataInputStream in,
            double[][] matrix
    ) throws IOException {
        int rows = in.readInt();

        if (rows != matrix.length) {
            throw new IOException(
                    "Matrix row mismatch."
            );
        }

        for (int i = 0; i < rows; i++) {
            int cols = in.readInt();

            if (cols != matrix[i].length) {
                throw new IOException(
                        "Matrix column mismatch."
                );
            }

            for (int j = 0; j < cols; j++) {
                matrix[i][j] = in.readDouble();
            }
        }
    }

    private static void writeArray(
            DataOutputStream out,
            double[] array
    ) throws IOException {
        out.writeInt(array.length);

        for (double value : array) {
            out.writeDouble(value);
        }
    }

    private static void readArray(
            DataInputStream in,
            double[] array
    ) throws IOException {
        int length = in.readInt();

        if (length != array.length) {
            throw new IOException(
                    "Array length mismatch."
            );
        }

        for (int i = 0; i < length; i++) {
            array[i] = in.readDouble();
        }
    }

    private static double clamp(double value) {
        return Math.max(
                -1.0,
                Math.min(1.0, value)
        );
    }

    private static float wrapDegrees(float degrees) {
        while (degrees >= 180.0f) {
            degrees -= 360.0f;
        }

        while (degrees < -180.0f) {
            degrees += 360.0f;
        }

        return degrees;
    }
}
