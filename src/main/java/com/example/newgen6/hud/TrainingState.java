package com.example.newgen6.hud;

/**
 * Lightweight immutable snapshot for the HUD.
 * No Minecraft or rendering logic belongs here.
 */
public record TrainingState(
        boolean aiEnabled,
        boolean training,
        long environmentSteps,
        long updates,
        float episodeReward,
        float meanReward,
        float policyLoss,
        float valueLoss,
        float entropy,
        float approxKl,
        float gradientNorm,
        float fps
) {
    public static TrainingState empty() {
        return new TrainingState(
                false,
                false,
                0L,
                0L,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f
        );
    }
}