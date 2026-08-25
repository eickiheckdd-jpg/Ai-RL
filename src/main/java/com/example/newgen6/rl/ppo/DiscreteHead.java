package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.nn.Dense;

import java.util.Random;

/**
 * One discrete action-group head: trunkOutput -> Dense -> logits -> softmax probs.
 *
 * Because the total log-prob of a factorized multi-discrete policy is a SUM
 * of per-group log-probs, each head's gradient only depends on its own
 * logits -- so this class can own the full sample/logProb/entropy/backward
 * cycle independently, and ActorCriticNetwork just sums the resulting
 * gradients-w.r.t.-trunk-output across heads.
 */
public class DiscreteHead {

    public final Dense dense;
    private float[] lastProbs;

    public DiscreteHead(int trunkSize, int numActions, Random rng) {
        this.dense = new Dense(trunkSize, numActions, rng);
    }

    /** Forward pass; caches the resulting probability distribution for sample()/logProb()/entropy(). */
    public float[] probs(float[] trunkOutput) {
        float[] logits = dense.forward(trunkOutput);
        this.lastProbs = softmax(logits);
        return lastProbs;
    }

    public int sample(Random rng) {
        float r = rng.nextFloat();
        float cum = 0f;
        for (int i = 0; i < lastProbs.length; i++) {
            cum += lastProbs[i];
            if (r <= cum) return i;
        }
        return lastProbs.length - 1;
    }

    public float logProb(int action) {
        return (float) Math.log(Math.max(lastProbs[action], 1e-8f));
    }

    public float entropy() {
        float h = 0f;
        for (float p : lastProbs) {
            if (p > 1e-8f) h -= p * (float) Math.log(p);
        }
        return h;
    }

    /**
     * Backprops the PPO policy-gradient term plus an entropy bonus through
     * this head's logits, and returns dLoss/dTrunkOutput.
     *
     * coeffPolicy == dLoss/dNewLogProb, precomputed once in ActorCriticNetwork
     * from the clipped-surrogate branch logic (shared across all heads,
     * since newLogProb = sum of per-group log-probs).
     *
     *   dLossPolicy/dLogit_j  = coeffPolicy * (1[j==action] - p_j)
     *   dLossEntropy/dLogit_j = -entropyCoef * p_j * (H - log p_j)
     */
    public float[] backward(int action, float coeffPolicy, float entropyCoef) {
        float[] gradLogits = new float[lastProbs.length];
        float h = entropy();
        for (int j = 0; j < lastProbs.length; j++) {
            float pj = lastProbs[j];
            float indicator = (j == action) ? 1f : 0f;
            float gradPolicy = coeffPolicy * (indicator - pj);
            float logPj = (float) Math.log(Math.max(pj, 1e-8f));
            float gradEntropy = -entropyCoef * pj * (h - logPj);
            gradLogits[j] = gradPolicy + gradEntropy;
        }
        return dense.backward(gradLogits);
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float[] out = new float[logits.length];
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            out[i] = (float) Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }
}
