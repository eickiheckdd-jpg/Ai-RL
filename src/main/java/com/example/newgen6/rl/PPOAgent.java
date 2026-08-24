package com.example.newgen6.rl;

import java.io.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class PPOAgent {
    public MLP actor, critic;
    private final int batchSize, stateDim, actionDim;

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
        if (isTraining.get()) return;
        actor.forward(state, outActions);
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
        // Gradient update calculation logic goes here
    }

    public void saveModel(String path) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(actor);
            out.writeObject(critic);
            System.out.println("[PPO] Saved model to " + path);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadModel(String path) {
        File f = new File(path);
        if (!f.exists()) {
            System.out.println("[PPO] File not found. Creating initial " + path);
            saveModel(path);
            return;
        }
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            actor = (MLP) in.readObject();
            critic = (MLP) in.readObject();
            actor.initTransient();
            critic.initTransient();
            System.out.println("[PPO] Loaded model from " + path);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}