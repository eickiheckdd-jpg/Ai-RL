package com.example.newgen6;

import java.util.Random;

public class DoubleDQNAgent {
    private final JavaMLP onlineNetwork;
    private final JavaMLP targetNetwork;
    private final ReplayBuffer replayBuffer;

    private final int actionCount;
    private final float gamma = 0.99f;
    private float epsilon = 1.0f;
    private final float epsilonMin = 0.05f;
    private final float epsilonDecay = 0.9995f; // Faster decay so it learns to aim sooner
    private final int batchSize = 16;           // Reduced so it doesn't lag your CPU
    private final int targetUpdateFrequency = 1000;
    private final int warmupSize = 200;         // Starts training much earlier

    private int trainingStepCount = 0;
    private int gameStepCount = 0;
    private final Random random = new Random();

    public DoubleDQNAgent(int inputSize, int actionCount) {
        this.actionCount = actionCount;
        this.onlineNetwork = new JavaMLP(inputSize, 128, 128, actionCount, 1e-4f);
        this.targetNetwork = new JavaMLP(inputSize, 128, 128, actionCount, 1e-4f);
        this.targetNetwork.copyWeightsFrom(this.onlineNetwork);
        this.replayBuffer = new ReplayBuffer(100000);
    }

    // Synchronized blocks main thread briefly to prevent reading corrupted weights mid-update
    public synchronized int selectAction(float[] state, boolean evaluationMode) {
        if (!evaluationMode && random.nextFloat() < epsilon) {
            return random.nextInt(actionCount);
        }

        float[] qValues = onlineNetwork.forward(state);
        int bestAction = 0;
        float maxQ = -Float.MAX_VALUE;
        for (int i = 0; i < qValues.length; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    // Instantly pushes to buffer on the main thread (zero lag)
    public void addExperience(float[] state, int action, float reward, float[] nextState, boolean done) {
        replayBuffer.add(state, action, reward, nextState, done);
        gameStepCount++;
    }

    // Runs on background thread
    public synchronized void trainBatch() {
        if (replayBuffer.size() < warmupSize) return;

        Transition[] batch = replayBuffer.sample(batchSize);

        for (Transition t : batch) {
            float[] qOnline = onlineNetwork.forward(t.state).clone();
            float targetQ = t.reward;

            if (!t.done) {
                float[] nextQOnline = onlineNetwork.forward(t.nextState);
                int bestNextAction = 0;
                float maxNextQOnline = -Float.MAX_VALUE;
                for (int i = 0; i < actionCount; i++) {
                    if (nextQOnline[i] > maxNextQOnline) {
                        maxNextQOnline = nextQOnline[i];
                        bestNextAction = i;
                    }
                }

                float[] nextQTarget = targetNetwork.forward(t.nextState);
                targetQ += gamma * nextQTarget[bestNextAction];
            }

            qOnline[t.action] = targetQ;
            onlineNetwork.trainStep(t.state, qOnline, 1.0f);
        }

        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }

        trainingStepCount++;
        if (trainingStepCount % targetUpdateFrequency == 0) {
            targetNetwork.copyWeightsFrom(onlineNetwork);
        }
    }

        public synchronized JavaMLP getOnlineNetwork() { return onlineNetwork; }
    public float getEpsilon() { return epsilon; }
    public void setEpsilon(float epsilon) { this.epsilon = epsilon; }
    public int getTrainingStepCount() { return trainingStepCount; }
    public int getGameStepCount() { return gameStepCount; }
    public void setTrainingStepCount(int stepCount) { this.trainingStepCount = stepCount; }
    
    // Added for ModelSerializer compatibility
    public int getStepCount() { return trainingStepCount; }
    public void setStepCount(int stepCount) { this.trainingStepCount = stepCount; }
}
