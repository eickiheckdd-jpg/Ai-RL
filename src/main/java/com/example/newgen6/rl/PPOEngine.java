package com.example.newgen6.rl;

import java.util.Random;
import java.util.concurrent.CompletableFuture;

public class PPOEngine {
    private final int stateDim = 8, actionDim = 5;
    private final float[] weightsActor = new float[stateDim * actionDim];
    private final float[] weightsCritic = new float[stateDim];
    private final Random rand = new Random();
    public float stdev = 0.20f;

    public PPOEngine() {
        for (int i = 0; i < weightsActor.length; i++) weightsActor[i] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weightsCritic.length; i++) weightsCritic[i] = (rand.nextFloat() - 0.5f) * 0.1f;
    }

    public void selectAction(float[] state, float[] outActions) {
        for (int i = 0; i < actionDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < stateDim; j++) sum += state[j] * weightsActor[i * stateDim + j];
            float mean = (float) Math.tanh(sum); // Bounded continuous output [-1, 1]
            float noise = (float) (rand.nextGaussian() * stdev);
            outActions[i] = Math.max(-1.0f, Math.min(1.0f, mean + noise));
        }
    }

    public void trainAsync(float[] state, float[] actions, float reward, float[] nextState) {
        CompletableFuture.runAsync(() -> {
            float vCurrent = 0.0f, vNext = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                vCurrent += state[i] * weightsCritic[i];
                vNext += nextState[i] * weightsCritic[i];
            }
            float advantage = reward + (0.99f * vNext) - vCurrent;

            // Gradient updates with strict clipping [-1.0, 1.0]
            for (int i = 0; i < stateDim; i++) {
                float gradC = Math.max(-1.0f, Math.min(1.0f, advantage * state[i]));
                weightsCritic[i] += 0.005f * gradC;
            }
            for (int i = 0; i < actionDim; i++) {
                for (int j = 0; j < stateDim; j++) {
                    float gradA = Math.max(-1.0f, Math.min(1.0f, advantage * actions[i] * state[j]));
                    weightsActor[i * stateDim + j] += 0.001f * gradA;
                }
            }
            if (stdev > 0.05f) stdev *= 0.9999f; // Gradual exploration decay
        });
    }
}
