package com.example.newgen6.rl;

/**
 * Hyperparameters tuned for pure-Java PPO under ~2 GB total RAM
 * and HT3 Sword curriculum.
 */
public final class Config {
    private Config() {}

    // Network
    public static final int OBS_DIM = 229;
    public static final int HIDDEN1 = 192;
    public static final int HIDDEN2 = 128;
    public static final int ACTION_DIM = 48;          // discrete hybrid actions
    public static final int LOOK_DIM = 2;             // continuous yaw/pitch deltas

    // PPO
    public static final float GAMMA = 0.99f;
    public static final float GAE_LAMBDA = 0.95f;
    public static final float CLIP_EPS = 0.15f;
    public static final float ENTROPY_COEF_START = 0.04f;
    public static final float ENTROPY_COEF_END = 0.005f;
    public static final float VALUE_COEF = 0.5f;
    public static final float MAX_GRAD_NORM = 0.5f;
    public static final float LEARNING_RATE = 2.5e-4f;
    public static final int UPDATE_EPOCHS = 4;
    public static final int MINI_BATCH = 64;

    // Rollout
    public static final int ROLLOUT_STEPS = 256;      // keep short for RAM
    public static final int MAX_EPISODE_STEPS = 1200; // ~60s at 20 TPS

    // Curriculum
    public static final int STAGES = 7;
    public static final int[] STAGE_THRESHOLDS = {0, 8000, 25000, 60000, 120000, 250000, 500000};

    // Exploration / speed control
    public static final float INITIAL_NOISE = 0.35f;  // high → shaking early
    public static final float FINAL_NOISE = 0.04f;
    public static final float LOOK_SCALE_START = 8.0f; // degrees per action early
    public static final float LOOK_SCALE_END = 2.2f;   // precise aiming later

    // Reward scaling
    public static final float REWARD_SCALE = 0.12f;
}
