package com.example.newgen6.rl;

/** Cycled via the 'C' keybind: OFF -> INFERENCE -> TRAINING -> OFF ... */
public enum AIState {
    OFF,
    INFERENCE,   // greedy playback, no exploration, no learning
    TRAINING;    // exploratory (sampled) actions, fills rollout buffer, triggers PPO updates

    public AIState next() {
        return switch (this) {
            case OFF -> INFERENCE;
            case INFERENCE -> TRAINING;
            case TRAINING -> OFF;
        };
    }
}
