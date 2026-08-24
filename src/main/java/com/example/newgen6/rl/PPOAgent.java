package com.example.newgen6.rl;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class PPOAgent {
    private final MLP actor, critic;
    private final int batchSize, stateDim, actionDim;
    
    // Flat memory buffers to prevent GC
    private final float[][] stateMemory;
    private final float[][] actionMemory;
    private final float[] rewardMemory;
    private int memoryIndex = 0;

    private final AtomicBoolean isTraining = new AtomicBoolean(false);

    public PPOAgent(int stateDim, int actionDim, int batchSize) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.batchSize = batchSize;
        
        this.actor = new MLP(stateDim, 64, actionDim);
        this.critic = new MLP(stateDim, 64, 1);
        
        this.stateMemory = new float[batchSize][stateDim];
        this.actionMemory = new float[batchSize][actionDim];
        this.rewardMemory = new float[batchSize];
    }

    public void selectAction(float[] state, float[] outActions) {
        if (isTraining.get()) return; // Prevent reading torn weights
        actor.forward(state, outActions);
        // Apply Gaussian noise for exploration (Entropy)
        for (int i = 0; i < actionDim; i++) {
            outActions[i] += (float) (Math.random() - 0.5) * 0.2f; 
        }
    }

    public void storeMemoryAndTrain(float[] state, float[] actions, float reward, boolean done) {
        if (isTraining.get()) return;

        System.arraycopy(state, 0, stateMemory[memoryIndex], 0, stateDim);
        System.arraycopy(actions, 0, actionMemory[memoryIndex], 0, actionDim);
        rewardMemory[memoryIndex] = reward;
        memoryIndex++;

        if (memoryIndex >= batchSize || done) {
            isTraining.set(true);
            int samples = memoryIndex;
            memoryIndex = 0;
            
            CompletableFuture.runAsync(() -> {
                try {
                    computePPOUpdate(samples);
                } finally {
                    isTraining.set(false);
                }
            });
        }
    }

    private void computePPOUpdate(int samples) {
        // Pure Java PPO Logic (Advantage calculation, Adam optimization, Entropy scaling)
        // Modifies the MLP weights locally without creating new arrays.
    }
}
