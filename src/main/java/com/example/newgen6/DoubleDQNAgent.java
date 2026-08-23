package com.example.newgen6;

import java.util.Random;

public class DoubleDQNAgent {
    private final int stateSize;
    private final int actionSize;
    
    // Public to allow ModelSerializer to access weights
    public final JavaMLP qNetwork;
    public final JavaMLP targetNetwork;
    private final ReplayBuffer memory;

    // Hyperparameters
    private float epsilon = 1.0f;
    private final float epsilonMin = 0.05f;
    private final float epsilonDecay = 0.9995f;
    private final float gamma = 0.99f;
    private final int batchSize = 64;
    private final int targetUpdateFreq = 1000;
    
    private int trainingStepCount = 0;
    private final Random random = new Random();

    public DoubleDQNAgent(int stateSize, int actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        // Neural Network architecture (Input -> 64 -> 64 -> Output)
        this.qNetwork = new JavaMLP(stateSize, 64, 64, actionSize, 0.0005f);
        this.targetNetwork = new JavaMLP(stateSize, 64, 64, actionSize, 0.0005f);
        this.targetNetwork.copyWeightsFrom(qNetwork);

        // Uses the upgraded zero-allocation buffer 
        this.memory = new ReplayBuffer(10000, stateSize);
    }

    public int selectAction(float[] state, boolean evaluationMode) {
        // Epsilon-greedy exploration
        if (!evaluationMode && random.nextFloat() < epsilon) {
            return random.nextInt(actionSize);
        }
        
        float[] qValues = qNetwork.forward(state);
        int bestAction = 0;
        float maxQ = -Float.MAX_VALUE;
        
        for (int i = 0; i < actionSize; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    public void addExperience(float[] state, int action, float reward, float[] nextState, boolean done) {
        memory.add(state, action, reward, nextState, done);
    }

    public void trainBatch() {
        if (memory.size() < batchSize) return;

        Transition[] batch = memory.sample(batchSize);
        
        // Prepare arrays for the vectorized batch pass
        float[][] batchInputs = new float[batchSize][stateSize];
        float[][] batchTargets = new float[batchSize][actionSize];

        for (int i = 0; i < batchSize; i++) {
            Transition t = batch[i];
            
            // Fast array copy for inputs
            System.arraycopy(t.state, 0, batchInputs[i], 0, stateSize);
            
            // Get current predictions to keep non-target action values unchanged (error = 0)
            float[] currentQ = qNetwork.forward(t.state);
            System.arraycopy(currentQ, 0, batchTargets[i], 0, actionSize);

            float targetValue = t.reward;
            
            if (!t.done) {
                // Double DQN Logic Step 1: Select the best next action using the MAIN network
                float[] nextQMain = qNetwork.forward(t.nextState);
                int bestNextAction = 0;
                float maxNextQMain = -Float.MAX_VALUE;
                for (int a = 0; a < actionSize; a++) {
                    if (nextQMain[a] > maxNextQMain) {
                        maxNextQMain = nextQMain[a];
                        bestNextAction = a;
                    }
                }
                
                // Double DQN Logic Step 2: Evaluate that specific action using the TARGET network
                float[] nextQTarget = targetNetwork.forward(t.nextState);
                targetValue += gamma * nextQTarget[bestNextAction];
            }

            // Update the specific action's target value
            batchTargets[i][t.action] = targetValue;
        }

        // Execute true mini-batch training (Massive stability upgrade over looping trainStep)
        qNetwork.trainBatchVectorized(batchInputs, batchTargets, 1.0f);

        // Decay exploration rate
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }

        // Sync target network periodically
        trainingStepCount++;
        if (trainingStepCount % targetUpdateFreq == 0) {
            targetNetwork.copyWeightsFrom(qNetwork);
        }
    }

    // Getters and Setters used by ModelSerializer and HUD
    public float getEpsilon() { 
        return epsilon; 
    }
    
    public void setEpsilon(float e) { 
        this.epsilon = e; 
    }
    
    public int getTrainingStepCount() { 
        return trainingStepCount; 
    }
    
    public void setTrainingStepCount(int count) { 
        this.trainingStepCount = count; 
    }
}
