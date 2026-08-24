package com.example.newgen6.rl;

import java.io.*;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

public class PPOEngine {
    private final int stateDim = 13, actionDim = 5; // Upgraded to 13 states
    private final float[] weightsActor = new float[stateDim * actionDim];
    private final float[] weightsCritic = new float[stateDim];
    private final Random rand = new Random();
    public float stdev = 0.25f;

    private final File primarySave = new File("pvp_brain.bin");
    private final File backupSave = new File("pvp_brain_backup.bin");

    public PPOEngine() {
        for (int i = 0; i < weightsActor.length; i++) weightsActor[i] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weightsCritic.length; i++) weightsCritic[i] = (rand.nextFloat() - 0.5f) * 0.1f;
        loadBrain();
        Runtime.getRuntime().addShutdownHook(new Thread(this::saveBrain));
    }

    public void selectAction(float[] state, float[] outActions) {
        for (int i = 0; i < actionDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < stateDim; j++) sum += state[j] * weightsActor[i * stateDim + j];
            float mean = (float) Math.tanh(sum); 
            
            // Dynamic exploration noise scales down when crosshair is locked
            float adaptiveNoise = stdev * (1.1f - Math.abs(state[3])); 
            outActions[i] = Math.max(-1.0f, Math.min(1.0f, mean + (float) (rand.nextGaussian() * adaptiveNoise)));
        }
    }

    // Thread-safe Async PPO Update Loop
    public void trainAsync(float[] stateIn, float[] actionsIn, float reward, float[] nextStateIn) {
        // Safe cloning prevents main-thread mutation during async computation
        final float[] state = stateIn.clone();
        final float[] actions = actionsIn.clone();
        final float[] nextState = nextStateIn.clone();

        CompletableFuture.runAsync(() -> {
            float vCurrent = 0.0f, vNext = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                vCurrent += state[i] * weightsCritic[i];
                vNext += nextState[i] * weightsCritic[i];
            }

            float advantage = reward + (0.98f * vNext) - vCurrent;

            // Critic Update
            for (int i = 0; i < stateDim; i++) {
                float gradC = Math.max(-1.0f, Math.min(1.0f, advantage * state[i]));
                weightsCritic[i] += 0.005f * gradC;
            }

            // Actor Update with Clipped Gradients
            for (int i = 0; i < actionDim; i++) {
                for (int j = 0; j < stateDim; j++) {
                    float policyGrad = advantage * actions[i] * state[j];
                    float clippedGrad = Math.max(-0.2f, Math.min(0.2f, policyGrad));
                    weightsActor[i * stateDim + j] += 0.002f * clippedGrad;
                }
            }

            if (stdev > 0.02f) stdev *= 0.99995f;
        });
    }

    public void saveBrainAsync() { CompletableFuture.runAsync(this::saveBrain); }

    public synchronized void saveBrain() {
        try {
            if (primarySave.exists()) {
                if (backupSave.exists()) backupSave.delete();
                primarySave.renameTo(backupSave);
            }
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(primarySave))) {
                for (float w : weightsActor) dos.writeFloat(w);
                for (float w : weightsCritic) dos.writeFloat(w);
                dos.writeFloat(stdev);
            }
        } catch (IOException ignored) {}
    }

    public synchronized void loadBrain() {
        File target = primarySave.exists() ? primarySave : (backupSave.exists() ? backupSave : null);
        if (target == null) return;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(target))) {
            for (int i = 0; i < weightsActor.length; i++) weightsActor[i] = dis.readFloat();
            for (int i = 0; i < weightsCritic.length; i++) weightsCritic[i] = dis.readFloat();
            stdev = dis.readFloat();
        } catch (IOException e) {
            if (target.equals(primarySave) && backupSave.exists()) loadBrain();
        }
    }
}
