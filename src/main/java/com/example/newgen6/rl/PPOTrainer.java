package com.example.newgen6.rl;

import java.util.List;

public class PPOTrainer {
    private final PolicyNetwork policy;
    private final float learningRate = 0.0001f;
    private final float gamma = 0.99f;
    private final float gaeLambda = 0.95f;

    public PPOTrainer(PolicyNetwork policy) {
        this.policy = policy;
    }

    public void train(RolloutBuffer buffer) {
        List<RolloutBuffer.Transition> transitions = buffer.getBatch();
        int batchSize = transitions.size();
        if (batchSize == 0) return;

        float[] advantages = new float[batchSize];
        float lastAdv = 0.0f;
        for (int t = batchSize - 1; t >= 0; t--) {
            RolloutBuffer.Transition tr = transitions.get(t);
            float nextValue = 0.0f;
            float delta = tr.reward + (gamma * nextValue);
            advantages[t] = delta + gamma * gaeLambda * lastAdv;
            lastAdv = advantages[t];
        }

        float mean = 0.0f;
        for (float a : advantages) mean += a;
        mean /= batchSize;

        float variance = 0.0f;
        for (float a : advantages) variance += (a - mean) * (a - mean);
        float std = (float) Math.sqrt(variance / batchSize + 1e-8f);

        for (int i = 0; i < batchSize; i++) {
            advantages[i] = (advantages[i] - mean) / std;
        }

        float[][] wTrunk = policy.getWTrunk();
        float[][] wMove = policy.getWMove();
        float[][] wYaw = policy.getWYaw();
        float[][] wPitch = policy.getWPitch();

        for (int t = 0; t < batchSize; t++) {
            RolloutBuffer.Transition tr = transitions.get(t);
            float adv = advantages[t];

            policy.forward(tr.state);

            updateCategoricalHead(wMove, policy.moveProbs, tr.moveAction, adv);
            updateCategoricalHead(wYaw, policy.yawProbs, tr.yawAction, adv);
            updateCategoricalHead(wPitch, policy.pitchProbs, tr.pitchAction, adv);

            for (int i = 0; i < Math.min(PolicyNetwork.INPUT_SIZE, tr.state.length); i += 10) { 
                for (int j = 0; j < PolicyNetwork.TRUNK_SIZE; j++) {
                    wTrunk[i][j] += learningRate * adv * tr.state[i];
                }
            }
        }
        buffer.clear();
    }

    private void updateCategoricalHead(float[][] weights, float[] probs, int targetBin, float adv) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                float grad = ((j == targetBin ? 1.0f : 0.0f) - probs[j]) * adv;
                weights[i][j] += learningRate * grad;
            }
        }
    }
}
