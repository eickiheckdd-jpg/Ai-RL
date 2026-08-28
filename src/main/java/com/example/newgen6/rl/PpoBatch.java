package com.example.newgen6.rl;

final class PpoBatch {
    int count;

    float[] obs;
    float[] actions;
    float[] oldLogProbs;
    float[] rewards;
    float[] values;
    float[] dones;

    float[] advantages;
    float[] returns;

    float lastValue;

    void computeAdvantages(float gamma, float lambda) {
        advantages = new float[count];
        returns = new float[count];

        float nextValue = lastValue;
        float lastAdv = 0.0f;

        for (int t = count - 1; t >= 0; t--) {
            boolean done = dones[t] > 0.5f;
            float nextV = done ? 0.0f : nextValue;
            float delta = rewards[t] + gamma * nextV - values[t];
            float adv = delta + gamma * lambda * (done ? 0.0f : 1.0f) * lastAdv;

            advantages[t] = adv;
            returns[t] = adv + values[t];

            nextValue = values[t];
            lastAdv = done ? 0.0f : adv;
        }

        // Normalize advantages
        float mean = 0.0f;
        for (int i = 0; i < count; i++) mean += advantages[i];
        mean /= Math.max(1, count);

        float var = 0.0f;
        for (int i = 0; i < count; i++) {
            float d = advantages[i] - mean;
            var += d * d;
        }
        var /= Math.max(1, count);
        float std = (float) Math.sqrt(var + 1e-8f);

        if (std > 1e-6f) {
            for (int i = 0; i < count; i++) {
                advantages[i] = (advantages[i] - mean) / std;
            }
        } else {
            for (int i = 0; i < count; i++) advantages[i] = 0.0f;
        }
    }
}