package com.example.newgen6.rl;

/**
 * Single source of truth for NEWGEN6 v2 shapes.
 * Minecraft Java 1.21.11 / Fabric.
 */
public final class RLConstants {
    private RLConstants() {}

    public static final int OBSERVATION_SIZE = 229;
    public static final int CONTEXT_TICKS = 200; // ~10.0s @ 20 TPS

    public static final int YAW_BUCKETS = 19;
    public static final int PITCH_BUCKETS = 17;
    public static final int AIM_JOINT = YAW_BUCKETS * PITCH_BUCKETS; // 323 (UI reference)

    public static final int MOVE_ACTIONS = 9; // 8 dirs + HOLD
    public static final int MOVE_HOLD = 8;

    public static final int FRAME_ENC = 96;
    public static final int TEMPORAL_FRAMES = 8; // recent frames fully encoded
    public static final int TRUNK_SIZE = 128;
    public static final int HIDDEN = 192;

    // Observation block bases (must match OBSERVATION_ABI_229.md)
    public static final int SELF_BASE = 0;
    public static final int TARGET_BASE = 32;
    public static final int NEARBY_BASE = 64;
    public static final int TIMING_BASE = 128;
    public static final int ITEM_BASE = 144;
    public static final int TERRAIN_BASE = 176;
    public static final int MISC_BASE = 208;

    public static final int NEARBY_SLOTS = 3;
    public static final int NEARBY_STRIDE = 21;

    public static final float TARGET_RANGE = 64.0f;
    public static final float MELEE_RANGE = 3.5f;

    public static final float MAX_YAW_DELTA_DEG = 30.0f;
    public static final float MAX_PITCH_DELTA_DEG = 20.0f;

    /**
     * Aim-only reward (train phase 1). Hardened against oscillation / switch exploits:
     *   r = ABS_WEIGHT * (-aimErr) + DELTA_WEIGHT * (prevErr - aimErr)
     * Delta is applied only when the same target entity continues.
     */
    public static final float REWARD_ABS_WEIGHT = 0.05f;
    public static final float REWARD_DELTA_WEIGHT = 0.5f;
    public static final float REWARD_NO_TARGET = -0.02f;
    public static final float REWARD_TARGET_SWITCH = -0.01f; // absolute-only tick, mild

    public static final int PPO_ROLLOUT = 256;
    public static final int PPO_EPOCHS = 4;
    public static final int PPO_MINIBATCH = 64;
    public static final float PPO_CLIP = 0.2f;
    public static final float GAMMA = 0.99f;
    public static final float GAE_LAMBDA = 0.95f;
    public static final float ENTROPY_COEF = 0.01f;
    public static final float VALUE_COEF = 0.5f;
    public static final float LR = 3e-4f;
    public static final float MAX_GRAD_NORM = 1.0f;

    public static void assertObsSize(int n) {
        if (n != OBSERVATION_SIZE) {
            throw new IllegalStateException(
                    "Observation size mismatch: got " + n + " expected " + OBSERVATION_SIZE);
        }
    }
}
