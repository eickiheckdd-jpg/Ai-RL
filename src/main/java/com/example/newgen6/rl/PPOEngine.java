package com.example.newgen6.rl;

import java.io.*;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

public class PPOEngine {
    private final int stateDim = 8, actionDim = 5;
    private final float[] weightsActor = new float[stateDim * actionDim];
    private final float[] weightsCritic = new float[stateDim];
    private final Random rand = new Random();
    public float stdev = 0.20f;

    private final File primarySave = new File("pvp_brain.bin");
    private final File backupSave = new File("pvp_brain_backup.bin");

    public PPOEngine() {
        for (int i = 0; i < weightsActor.length; i++) weightsActor[i] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weightsCritic.length; i++) weightsCritic[i] = (rand.nextFloat() - 0.5f) * 0.1f;

        // Auto-load brain on startup
        loadBrain();

        // Safety save hook for unexpected game closes
        Runtime.getRuntime().addShutdownHook(new Thread(this::saveBrain));
    }

    public void selectAction(float[] state, float[] outActions) {
        for (int i = 0; i < actionDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < stateDim; j++) sum += state[j] * weightsActor[i * stateDim + j];
            float mean = (float) Math.tanh(sum);
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
            if (stdev > 0.05f) stdev *= 0.9999f;
        });
    }

    // Safely save model weights asynchronously
    public void saveBrainAsync() {
        CompletableFuture.runAsync(this::saveBrain);
    }

    public synchronized void saveBrain() {
        try {
            // Backup existing file before saving new data
            if (primarySave.exists()) {
                if (backupSave.exists()) backupSave.delete();
                primarySave.renameTo(backupSave);
            }

            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(primarySave))) {
                for (float w : weightsActor) dos.writeFloat(w);
                for (float w : weightsCritic) dos.writeFloat(w);
                dos.writeFloat(stdev);
                System.out.println("[PPO Engine] Auto-saved brain successfully!");
            }
        } catch (IOException e) {
            System.err.println("[PPO Engine] Auto-save failed: " + e.getMessage());
        }
    }

    public synchronized void loadBrain() {
        File target = primarySave.exists() ? primarySave : (backupSave.exists() ? backupSave : null);
        if (target == null) {
            System.out.println("[PPO Engine] No existing brain file found. Initializing fresh model.");
            return;
        }

        try (DataInputStream dis = new DataInputStream(new FileInputStream(target))) {
            for (int i = 0; i < weightsActor.length; i++) weightsActor[i] = dis.readFloat();
            for (int i = 0; i < weightsCritic.length; i++) weightsCritic[i] = dis.readFloat();
            stdev = dis.readFloat();
            System.out.println("[PPO Engine] Successfully loaded brain from " + target.getName());
        } catch (IOException e) {
            System.err.println("[PPO Engine] Corrupted brain file, switching to backup...");
            if (target.equals(primarySave) && backupSave.exists()) {
                primarySave.delete();
                loadBrain(); // Try loading backup
            }
        }
    }
}
