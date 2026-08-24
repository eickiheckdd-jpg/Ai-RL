package com.example.newgen6.rl;

import java.io.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public class PPOAgent implements Serializable {
    private static final long serialVersionUID = 1L;

    public MLP actor, critic;
    private final int batchSize, stateDim, actionDim;

    private final float[][] stateMemory;
    private final float[][] actionMemory;
    private final float[] rewardMemory;
    private final float[] logProbMemory;
    private final boolean[] doneMemory;
    private int memoryIndex = 0;

    private final AtomicBoolean isTraining = new AtomicBoolean(false);
    private static final float GAMMA = 0.99f;
    
    // Made public so HUD rendering can access EPSILON directly
    public static final float EPSILON = 0.2f;    // PPO Clip threshold
    
    // Dynamic noise settings (Saved to model state and made public for HUD)
    public float stdev = 0.3f;                    
    private static final float MIN_STDEV = 0.05f;  
    private static final float DECAY_RATE = 0.995f; 

    public PPOAgent(int stateDim, int actionDim, int batchSize) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.batchSize = batchSize;

        this.actor = new MLP(stateDim, 64, actionDim);
        this.critic = new MLP(stateDim, 64, 1);

        this.stateMemory = new float[batchSize][stateDim];
        this.actionMemory = new float[batchSize][actionDim];
        this.rewardMemory = new float[batchSize];
        this.logProbMemory = new float[batchSize];
        this.doneMemory = new boolean[batchSize];
    }

    public void selectAction(float[] state, float[] outActions) {
        actor.forward(state, outActions);
        for (int i = 0; i < actionDim; i++) {
            outActions[i] += (float) (Math.random() - 0.5) * (stdev * 2.0f);
        }
    }

    private float computeLogProb(float[] mean, float[] action) {
        float logProb = 0.0f;
        for (int i = 0; i < actionDim; i++) {
            float diff = action[i] - mean[i];
            logProb += -0.5f * ((diff * diff) / (stdev * stdev) + (float) Math.log(2 * Math.PI * stdev * stdev));
        }
        return logProb;
    }

    public void storeMemoryAndTrain(float[] state, float[] actions, float reward, boolean done) {
        System.arraycopy(state, 0, stateMemory[memoryIndex], 0, stateDim);
        System.arraycopy(actions, 0, actionMemory[memoryIndex], 0, actionDim);
        rewardMemory[memoryIndex] = reward;
        doneMemory[memoryIndex] = done;

        float[] mean = new float[actionDim];
        actor.forward(state, mean);
        logProbMemory[memoryIndex] = computeLogProb(mean, actions);

        memoryIndex++;

        if (memoryIndex >= batchSize || done) {
            if (isTraining.compareAndSet(false, true)) {
                int samples = memoryIndex;
                memoryIndex = 0;

                // Decay noise safely over time
                if (stdev > MIN_STDEV) {
                    stdev = Math.max(MIN_STDEV, stdev * DECAY_RATE);
                }

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
        float lrActor = 0.0003f;
        float lrCritic = 0.001f;
        int epochs = 4;

        float[] returns = new float[samples];
        float[] advantages = new float[samples];
        float[] criticVal = new float[1];

        // 1. Compute Discounted Future Returns & Advantage Estimation (TD)
        float runningReturn = 0.0f;
        float advMean = 0.0f;
        for (int t = samples - 1; t >= 0; t--) {
            if (doneMemory[t]) runningReturn = 0.0f;
            runningReturn = rewardMemory[t] + (GAMMA * runningReturn);
            returns[t] = runningReturn;

            critic.forward(stateMemory[t], criticVal);
            advantages[t] = returns[t] - criticVal[0];
            advMean += advantages[t];
        }

        // 2. Normalize Advantages (Prevents Explosive Gradients)
        advMean /= samples;
        float advStd = 0.0f;
        for (int t = 0; t < samples; t++) {
            float diff = advantages[t] - advMean;
            advStd += diff * diff;
        }
        advStd = (float) Math.sqrt(advStd / samples) + 1e-8f;

        for (int t = 0; t < samples; t++) {
            advantages[t] = (advantages[t] - advMean) / advStd;
        }

        // 3. PPO Multi-Epoch Optimization
        for (int e = 0; e < epochs; e++) {
            for (int k = 0; k < samples; k++) {
                float[] state = stateMemory[k];
                float[] action = actionMemory[k];
                float oldLogProb = logProbMemory[k];
                float advantage = advantages[k];

                // Actor Optimization with PPO Clipping
                float[] currentMean = new float[actionDim];
                float[][] actorActivations = actor.forwardDetailed(state, currentMean);
                float newLogProb = computeLogProb(currentMean, action);

                // Safety Clamp on Exponent to prevent NaN/Infinity
                float logRatio = Math.max(-20.0f, Math.min(20.0f, newLogProb - oldLogProb));
                float ratio = (float) Math.exp(logRatio);
                
                float surr1 = ratio * advantage;
                float surr2 = Math.max(Math.min(ratio, 1.0f + EPSILON), 1.0f - EPSILON) * advantage;
                
                float policyGrad = Math.min(surr1, surr2);
                float[] actorGradients = new float[actionDim];
                for (int i = 0; i < actionDim; i++) {
                    float diff = action[i] - currentMean[i];
                    actorGradients[i] = (diff / (stdev * stdev)) * policyGrad;
                }
                actor.trainBackward(actorActivations, actorGradients, lrActor);

                // Critic Optimization (MSE Loss Value Function)
                float[][] criticActivations = critic.forwardDetailed(state, criticVal);
                float criticError = returns[k] - criticVal[0];
                critic.trainBackward(criticActivations, new float[]{criticError}, lrCritic);
            }
        }
    }

    public void saveModel(String path) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(actor);
            out.writeObject(critic);
            out.writeFloat(stdev); // Persist exploration decay state
            System.out.println("[PPO] Saved model to " + path + " | STDEV: " + stdev);
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
            stdev = in.readFloat(); // Restore persisted exploration noise level
            actor.initTransient();
            critic.initTransient();
            System.out.println("[PPO] Loaded model from " + path + " | Restored STDEV: " + stdev);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}