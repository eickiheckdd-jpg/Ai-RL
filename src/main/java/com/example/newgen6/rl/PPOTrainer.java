package com.example.newgen6.rl;

public class PPOTrainer {
    private final PolicyNetwork policy;
    private final float gamma = 0.99f;
    private final float gaeLambda = 0.95f;
    private final float learningRate = 0.0003f;

    private float[] advantages;

    public PPOTrainer(PolicyNetwork policy) {
        this.policy = policy;
    }

    public void train(RolloutBuffer buffer) {
        if (buffer.size == 0) return;

        if (advantages == null || advantages.length < buffer.capacity) {
            advantages = new float[buffer.capacity];
        }

        // Calculate Advantages
        float lastAdv = 0.0f;
        for (int t = buffer.size - 1; t >= 0; t--) {
            float nextVal = (t == buffer.size - 1 || buffer.dones[t]) ? 0.0f : buffer.values[t + 1];
            float delta = buffer.rewards[t] + gamma * nextVal - buffer.values[t];
            advantages[t] = delta + gamma * gaeLambda * (buffer.dones[t] ? 0.0f : lastAdv);
            lastAdv = advantages[t];
        }

        // Apply Pure Java Weight Updates (Stochastic Gradient Descent approximation for slice)
        float[][] w2 = policy.getW2();
        for (int t = 0; t < buffer.size; t++) {
            float adv = advantages[t];
            float[] state = buffer.states[t];
            for (int i = 0; i < w2.length; i++) {
                for (int j = 0; j < w2[i].length; j++) {
                    w2[i][j] += learningRate * adv * state[i % state.length];
                }
            }
        }
        buffer.clear();
    }
}
