package com.example.newgen6.rl;

/**
 * Single source of truth for the NewGen6 RL architecture.
 *
 * Do not duplicate these architecture values in individual NN/PPO/environment
 * classes. Other classes should reference these constants.
 */
public final class RLConstants {

    private RLConstants() {
    }

    // ============================================================
    // Observation / temporal architecture
    // ============================================================

    public static final int OBSERVATION_SIZE = 229;

    /**
     * Number of Minecraft ticks represented by the recurrent context.
     *
     * At the normal 20 TPS tick rate this is approximately 10 seconds.
     */
    public static final int CONTEXT_TICKS = 200;

    // ============================================================
    // Neural-network architecture
    // ============================================================

    /**
     * GRU hidden-state width.
     *
     * Keep this centralized so the GRU, rollout buffer, policy network,
     * HUD, and training code cannot accidentally disagree.
     */
    public static final int GRU_HIDDEN_SIZE = 192;

    public static final int TRUNK_SIZE = 128;

    // ============================================================
    // Policy action heads
    // ============================================================

    public static final int MOVE_ACTIONS = 9;
    public static final int YAW_ACTIONS = 19;
    public static final int PITCH_ACTIONS = 17;

    public static final int JUMP_ACTIONS = 2;
    public static final int SPRINT_ACTIONS = 2;
    public static final int SNEAK_ACTIONS = 2;
    public static final int ATTACK_ACTIONS = 2;

    public static final int VALUE_OUTPUTS = 1;

    // ============================================================
    // PPO defaults
    // ============================================================

    public static final float GAMMA = 0.99f;
    public static final float GAE_LAMBDA = 0.95f;
    public static final float CLIP_EPSILON = 0.20f;

    public static final float VALUE_COEF = 0.50f;
    public static final float ENTROPY_COEF = 0.01f;

    public static final float MAX_GRAD_NORM = 0.50f;

    public static final int PPO_EPOCHS = 4;

    /**
     * Truncated-BPTT/PPO sequence length.
     *
     * This is deliberately separate from CONTEXT_TICKS.
     * CONTEXT_TICKS describes the agent's intended temporal context;
     * SEGMENT_LENGTH describes one training segment used during updates.
     */
    public static final int SEGMENT_LENGTH = 32;

    // ============================================================
    // Rollout defaults
    // ============================================================

    /**
     * Number of environment transitions collected before a PPO update.
     *
     * This is bounded to keep memory use practical on Pojav.
     */
    public static final int ROLLOUT_CAPACITY = 256;

    // ============================================================
    // Optimizer defaults
    // ============================================================

    public static final float LEARNING_RATE = 3.0e-4f;

    public static final float ADAM_BETA1 = 0.9f;
    public static final float ADAM_BETA2 = 0.999f;
    public static final float ADAM_EPSILON = 1.0e-8f;

    // ============================================================
    // Numerical safety
    // ============================================================

    public static final float MIN_PROBABILITY = 1.0e-12f;

    public static final float NORMALIZATION_EPSILON = 1.0e-8f;

    public static final float MAX_REASONABLE_OBSERVATION =
            1000000.0f;

    // ============================================================
    // Model metadata
    // ============================================================

    /**
     * Increase when the serialized parameter architecture changes.
     */
    public static final int ARCHITECTURE_VERSION = 1;

    /**
     * Increase when the binary model/checkpoint format changes.
     */
    public static final int MODEL_FORMAT_VERSION = 1;

    // ============================================================
    // Compile-time/runtime architecture validation
    // ============================================================

    /**
     * Performs consistency checks that do not require Minecraft classes.
     *
     * Call this during RL-system initialization.
     */
    public static void validateArchitecture() {
        requirePositive("OBSERVATION_SIZE", OBSERVATION_SIZE);
        requirePositive("CONTEXT_TICKS", CONTEXT_TICKS);

        requirePositive("GRU_HIDDEN_SIZE", GRU_HIDDEN_SIZE);
        requirePositive("TRUNK_SIZE", TRUNK_SIZE);

        requirePositive("MOVE_ACTIONS", MOVE_ACTIONS);
        requirePositive("YAW_ACTIONS", YAW_ACTIONS);
        requirePositive("PITCH_ACTIONS", PITCH_ACTIONS);
        requirePositive("JUMP_ACTIONS", JUMP_ACTIONS);
        requirePositive("SPRINT_ACTIONS", SPRINT_ACTIONS);
        requirePositive("SNEAK_ACTIONS", SNEAK_ACTIONS);
        requirePositive("ATTACK_ACTIONS", ATTACK_ACTIONS);

        requirePositive("VALUE_OUTPUTS", VALUE_OUTPUTS);

        requirePositive("PPO_EPOCHS", PPO_EPOCHS);
        requirePositive("SEGMENT_LENGTH", SEGMENT_LENGTH);
        requirePositive("ROLLOUT_CAPACITY", ROLLOUT_CAPACITY);

        if (!(GAMMA > 0.0f && GAMMA <= 1.0f)) {
            throw new IllegalStateException(
                    "GAMMA must be in (0, 1], got " + GAMMA
            );
        }

        if (!(GAE_LAMBDA >= 0.0f && GAE_LAMBDA <= 1.0f)) {
            throw new IllegalStateException(
                    "GAE_LAMBDA must be in [0, 1], got " + GAE_LAMBDA
            );
        }

        if (!(CLIP_EPSILON > 0.0f && CLIP_EPSILON < 1.0f)) {
            throw new IllegalStateException(
                    "CLIP_EPSILON must be in (0, 1), got " + CLIP_EPSILON
            );
        }

        if (!(LEARNING_RATE > 0.0f)) {
            throw new IllegalStateException(
                    "LEARNING_RATE must be > 0, got " + LEARNING_RATE
            );
        }

        if (!(MAX_GRAD_NORM > 0.0f)) {
            throw new IllegalStateException(
                    "MAX_GRAD_NORM must be > 0, got " + MAX_GRAD_NORM
            );
        }

        if (!(ADAM_BETA1 >= 0.0f && ADAM_BETA1 < 1.0f)) {
            throw new IllegalStateException(
                    "ADAM_BETA1 must be in [0, 1), got " + ADAM_BETA1
            );
        }

        if (!(ADAM_BETA2 >= 0.0f && ADAM_BETA2 < 1.0f)) {
            throw new IllegalStateException(
                    "ADAM_BETA2 must be in [0, 1), got " + ADAM_BETA2
            );
        }

        if (!(ADAM_EPSILON > 0.0f)) {
            throw new IllegalStateException(
                    "ADAM_EPSILON must be > 0, got " + ADAM_EPSILON
            );
        }

        if (SEGMENT_LENGTH > ROLLOUT_CAPACITY) {
            throw new IllegalStateException(
                    "SEGMENT_LENGTH (" + SEGMENT_LENGTH
                            + ") cannot exceed ROLLOUT_CAPACITY ("
                            + ROLLOUT_CAPACITY + ")"
            );
        }
    }

    private static void requirePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalStateException(
                    name + " must be > 0, got " + value
            );
        }
    }
}