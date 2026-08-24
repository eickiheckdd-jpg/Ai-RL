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
        // Fixed: Mouse controls do NOT stop when training starts
        actor.forward(state, outActions);
        for (int i = 0; i < actionDim; i++) {
            outActions[i] += (float) (Math.random() - 0.5) * 0.35f;
        }
    }

    public void storeMemoryAndTrain(float[] state, float[] actions, float reward, boolean done) {
        // Deep copy state and action values into separate slots to prevent array reference mutations
        System.arraycopy(state, 0, stateMemory[memoryIndex], 0, stateDim);
        System.arraycopy(actions, 0, actionMemory[memoryIndex], 0, actionDim);
        rewardMemory[memoryIndex] = reward;
        memoryIndex++;

        if (memoryIndex >= batchSize || done) {
            if (isTraining.compareAndSet(false, true)) {
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
    }

    private void computePPOUpdate(int samples) {
        float lr = 0.001f;
        float[] actorOut = new float[actionDim];
        float[] criticOut = new float[1];

        for (int k = 0; k < samples; k++) {
            float[] state = stateMemory[k];
            float[] action = actionMemory[k];
            float reward = rewardMemory[k];

            float[] actorH2 = actor.forward(state, actorOut);
            float[] criticH2 = critic.forward(state, criticOut);

            // TD Advantage Signal
            float advantage = reward - criticOut[0];

            // Update Critic Output Layer
            critic.weights3[0][0] += lr * advantage;
            for (int i = 0; i < criticH2.length; i++) {
                critic.weights3[i][0] += lr * advantage * criticH2[i];
            }

            // Update Actor Output Layer
            for (int j = 0; j < actionDim; j++) {
                float error = (action[j] - actorOut[j]) * advantage;
                for (int i = 0; i < actorH2.length; i++) {
                    actor.weights3[i][j] += lr * error * actorH2[i];
                }
            }
        }
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
