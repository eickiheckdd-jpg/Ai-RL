package com.example.newgen6.rl;

public final class AgentConfig {
    // Observation / memory
    public static final int OBS_DIM = 229;
    public static final int HISTORY_TICKS = 200;

    // Action space:
    // 0 forward, 1 backward, 2 left, 3 right,
    // 4 jump, 5 sprint, 6 sneak, 7 attack,
    // 8 mouseX, 9 mouseY
    public static final int KEY_ACTIONS = 8;
    public static final int CONT_ACTIONS = 2;
    public static final int ACTION_DIM = KEY_ACTIONS + CONT_ACTIONS;

    // PPO / rollout
    public static final int ROLLOUT_STEPS = 512;
    public static final float GAMMA = 0.99f;
    public static final float LAMBDA = 0.95f;
    public static final float CLIP_EPS = 0.2f;
    public static final float ENTROPY_COEF = 0.005f;
    public static final float LEARNING_RATE = 0.0003f;
    public static final int PPO_EPOCHS = 3;
    public static final int MINIBATCH_SIZE = 64;

    // Episode
    public static final int MAX_EPISODE_TICKS = 20 * 180;

    // Mouse safety
    public static final float MOUSE_CLAMP = 24.0f;

    // Checkpointing
    public static final int CHECKPOINT_VERSION = 2;
    public static final int CHECKPOINT_TICK_INTERVAL = 20 * 60;   // once per minute while training
    public static final int CHECKPOINT_UPDATE_INTERVAL = 2;       // every 2 PPO updates

    private AgentConfig() {
    }
}