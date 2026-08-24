package com.example.newgen6.rl;

import java.io.*;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

public class PPOEngine {
    private final int stateDim = 13, hiddenDim = 24, actionDim = 7;
    
    // Deep Neural Weights
    private final float[] wActor1 = new float[stateDim * hiddenDim];
    private final float[] wActor2 = new float[hiddenDim * actionDim];
    private final float[] wCritic1 = new float[stateDim * hiddenDim];
    private final float[] wCritic2 = new float[hiddenDim];

    private final Random rand = new Random();
    public float stdev = 0.25f;

    private float totalRewardTracker = 0.0f;
    private int trainingSteps = 0;

    private final File primarySave = new File("pvp_brain.bin");
    private final File backupSave = new File("pvp_brain_backup.bin");

    public PPOEngine() {
        initWeights(wActor1); initWeights(wActor2);
        initWeights(wCritic1); initWeights(wCritic2);
        loadBrain();
        Runtime.getRuntime().addShutdownHook(new Thread(this::saveBrain));
    }

    private void initWeights(float[] arr) {
        for (int i = 0; i < arr.length; i++) arr[i] = (rand.nextFloat() - 0.5f) * 0.1f;
    }

    // Forward Pass with ReLU Hidden Layer
    public void selectAction(float[] state, float[] outActions) {
        float[] hidden = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < stateDim; j++) sum += state[j] * wActor1[i * stateDim + j];
            hidden[i] = Math.max(0.0f, sum); // ReLU Activation
        }

        for (int i = 0; i < actionDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hiddenDim; j++) sum += hidden[j] * wActor2[i * hiddenDim + j];
            float mean = (float) Math.tanh(sum);
            // Dynamic exploration noise
            float noise = (float) (rand.nextGaussian() * (stdev * (1.1f - Math.abs(state[3]))));
            outActions[i] = Math.max(-1.0f, Math.min(1.0f, mean + noise));
        }
    }

    // Async Backpropagation & PPO Updates
    public void trainAsync(float[] stateIn, float[] actionsIn, float reward, float[] nextStateIn) {
        final float[] state = stateIn.clone();
        final float[] actions = actionsIn.clone();
        final float[] nextState = nextStateIn.clone();

        CompletableFuture.runAsync(() -> {
            // Live Progress Tracker
            totalRewardTracker += reward;
            trainingSteps++;
            if (trainingSteps % 1200 == 0) { // Logs progress every 1 minute
                System.out.printf("[PPO Engine] 1-Min Avg Reward: %.3f | Current Noise: %.4f%n", 
                                  (totalRewardTracker / 1200.0f), stdev);
                totalRewardTracker = 0.0f;
            }

            float[] hCurr = forwardHidden(state, wCritic1);
            float[] hNext = forwardHidden(nextState, wCritic1);

            float vCurrent = 0.0f, vNext = 0.0f;
            for (int i = 0; i < hiddenDim; i++) {
                vCurrent += hCurr[i] * wCritic2[i];
                vNext += hNext[i] * wCritic2[i];
            }

            float advantage = reward + (0.98f * vNext) - vCurrent;

            // Critic Weight Update
            for (int i = 0; i < hiddenDim; i++) {
                wCritic2[i] += 0.005f * Math.max(-1.0f, Math.min(1.0f, advantage * hCurr[i]));
                for (int j = 0; j < stateDim; j++) {
                    if (hCurr[i] > 0) wCritic1[i * stateDim + j] += 0.002f * advantage * state[j];
                }
            }

            // Actor Weight Update
            float[] hActor = forwardHidden(state, wActor1);
            for (int i = 0; i < actionDim; i++) {
                float grad = Math.max(-0.2f, Math.min(0.2f, advantage * actions[i]));
                for (int j = 0; j < hiddenDim; j++) {
                    wActor2[i * hiddenDim + j] += 0.002f * grad * hActor[j];
                    if (hActor[j] > 0) {
                        for (int k = 0; k < stateDim; k++) {
                            wActor1[j * stateDim + k] += 0.0005f * grad * state[k];
                        }
                    }
                }
            }

            if (stdev > 0.01f) stdev *= 0.99995f;
        });
    }

    private float[] forwardHidden(float[] in, float[] weights) {
        float[] h = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < in.length; j++) sum += in[j] * weights[i * in.length + j];
            h[i] = Math.max(0.0f, sum);
        }
        return h;
    }

    public void saveBrainAsync() { CompletableFuture.runAsync(this::saveBrain); }

    public synchronized void saveBrain() {
        try {
            if (primarySave.exists()) {
                if (backupSave.exists()) backupSave.delete();
                primarySave.renameTo(backupSave);
            }
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(primarySave))) {
                for (float w : wActor1) dos.writeFloat(w);
                for (float w : wActor2) dos.writeFloat(w);
                for (float w : wCritic1) dos.writeFloat(w);
                for (float w : wCritic2) dos.writeFloat(w);
                dos.writeFloat(stdev);
            }
        } catch (IOException ignored) {}
    }

    public synchronized void loadBrain() {
        File target = primarySave.exists() ? primarySave : (backupSave.exists() ? backupSave : null);
        if (target == null) return;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(target))) {
            for (int i = 0; i < wActor1.length; i++) wActor1[i] = dis.readFloat();
            for (int i = 0; i < wActor2.length; i++) wActor2[i] = dis.readFloat();
            for (int i = 0; i < wCritic1.length; i++) wCritic1[i] = dis.readFloat();
            for (int i = 0; i < wCritic2.length; i++) wCritic2[i] = dis.readFloat();
            stdev = dis.readFloat();
        } catch (IOException e) {
            if (target.equals(primarySave) && backupSave.exists()) loadBrain();
        }
    }
}
