package com.example.newgen6.rl.nn;

import java.util.Random;

/**
 * A categorical (softmax) distribution over N discrete logits.
 * Used for every PPO action head (move, yaw bucket, pitch bucket, and each
 * binary toggle encoded as a 2-way categorical).
 *
 * Numerically stable softmax:
 *   p_i = exp(logit_i - max(logits)) / sum_j exp(logit_j - max(logits))
 *
 * log-probability of a sampled action a:
 *   logP(a) = logit_a - max(logits) - log( sum_j exp(logit_j - max(logits)) )
 *
 * entropy:
 *   H = -sum_i p_i * log(p_i)
 */
public final class Categorical {

    /** Computes numerically-stable softmax probabilities from logits. Does not mutate input. */
    public static double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) if (l > max) max = l;
        double sum = 0.0;
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
    }

    /** Samples an action index from the categorical defined by logits. */
    public static int sample(double[] logits, Random rng) {
        double[] probs = softmax(logits);
        double u = rng.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (u <= cumulative) return i;
        }
        return probs.length - 1; // fallback for floating point edge case
    }

    /** Deterministic (argmax) action selection, used at evaluation time. */
    public static int argmax(double[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    /** log P(action) under the softmax distribution defined by logits. */
    public static double logProb(double[] logits, int action) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) if (l > max) max = l;
        double sumExp = 0.0;
        for (double l : logits) sumExp += Math.exp(l - max);
        double logSumExp = max + Math.log(sumExp);
        return logits[action] - logSumExp;
    }

    /** Shannon entropy of the softmax distribution defined by logits (nats). */
    public static double entropy(double[] logits) {
        double[] probs = softmax(logits);
        double h = 0.0;
        for (double p : probs) {
            if (p > 1e-12) h -= p * Math.log(p);
        }
        return h;
    }

    /**
     * Gradient of [-logP(action)] w.r.t. each logit, i.e. d(-logP)/d(logit_i) = p_i - 1{i==action}.
     * This is the standard softmax-cross-entropy gradient and is what PPO's
     * backward pass needs to push into the network (see PolicyValueNetwork#backward).
     */
    public static double[] dNegLogProbDLogits(double[] logits, int action) {
        double[] probs = softmax(logits);
        double[] grad = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            grad[i] = probs[i] - (i == action ? 1.0 : 0.0);
        }
        return grad;
    }

    private Categorical() {}
}
