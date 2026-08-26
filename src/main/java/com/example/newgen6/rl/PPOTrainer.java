package com.example.newgen6.rl;

public class PPOTrainer {
    private final PolicyNetwork policy;
    private final float gamma = 0.99f;
    private final float gaeLambda = 0.95f;
    private final float learningRate = 0.0003f;
    private final float clipRatio = 0.2f;

    // Pre-allocated arrays for advantage math
    private float[] advantages;
    private float[] returns;

    public PPOTrainer(PolicyNetwork policy) {
        this.policy = policy;
    }

    public void train(RolloutBuffer buffer) {
        if (buffer.size == 0) return;

        if (advantages == null || advantages.length < buffer.capacity) {
            advantages = new float[buffer.capacity];
            returns = new float[buffer.capacity];
        }

        // 1. Calculate Generalized Advantage Estimation (GAE)
        float lastAdvantage = 0.0f;
        for (int t = buffer.size - 1; t >= 0; t--) {
            float nextValue = (t == buffer.size - 1 || buffer.dones[t]) ? 0.0f : buffer.values[t + 1];
            float delta = buffer.rewards[t] + gamma * nextValue - buffer.values[t];
            
            advantages[t] = delta + gamma * gaeLambda * (buffer.dones[t] ? 0.0f : lastAdvantage);
            lastAdvantage = advantages[t];
            returns[t] = advantages[t] + buffer.values[t];
        }

        // 2. Normalize Advantages (Pure Java)
        float advMean = calculateMean(advantages, buffer.size);
        float advStd = calculateStd(advantages, buffer.size, advMean);
        for (int i = 0; i < buffer.size; i++) {
            advantages[i] = (advantages[i] - advMean) / (advStd + 1e-8f);
        }

        // 3. (Simplified) Gradient Descent Weight Update Step
        // A production PPO uses epochs and minibatches here.
        // For the slice, we apply a naive SGD step to policy weights using the advantages.
        for (int t = 0; t < buffer.size; t++) {
            float[] state = buffer.states[t];
            float adv = advantages[t];
            
            // Backpropagate advantage into weights (Simulated SGD for Policy)
            float[][] w2 = policy.getW2();
            for (int i = 0; i < w2.length; i++) {
                for (int j = 0; j < w2[i].length; j++) {
                    // Update weights pushing toward advantageous actions
                    w2[i][j] += learningRate * adv * state[i % state.length]; 
                }
            }
        }

        buffer.clear(); // Ready for next episode
    }

    private float calculateMean(float[] data, int size) {
        float sum = 0;
        for (int i = 0; i < size; i++) sum += data[i];
        return sum / size;
    }

    private float calculateStd(float[] data, int size, float mean) {
        float sumSq = 0;
        for (int i = 0; i < size; i++) sumSq += Math.pow(data[i] - mean, 2);
        return (float) Math.sqrt(sumSq / size);
    }
}
