package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.nn.Categorical;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

/**
 * Pure PPO/GAE mathematics.
 *
 * No Minecraft dependencies.
 */
public final class PPOMath {

    private PPOMath() {}

    /**
     * Computes generalized advantage estimates and returns.
     *
     * delta_t = r_t + gamma * V_{t+1} * (1-done_t) - V_t
     * A_t     = delta_t + gamma*lambda*(1-done_t)*A_{t+1}
     * R_t     = A_t + V_t
     */
    public static float[][] computeGAE(
            float[] rewards,
            float[] values,
            boolean[] dones,
            float bootstrapValue,
            float gamma,
            float gaeLambda) {

        if (rewards == null || values == null || dones == null) {
            throw new IllegalArgumentException("GAE arrays cannot be null");
        }

        if (rewards.length != values.length ||
                rewards.length != dones.length) {
            throw new IllegalArgumentException(
                    "GAE array lengths do not match"
            );
        }

        if (rewards.length == 0) {
            return new float[][]{
                    new float[0],
                    new float[0]
            };
        }

        if (!Float.isFinite(bootstrapValue)) {
            throw new IllegalArgumentException(
                    "bootstrapValue must be finite"
            );
        }

        float[] advantages = new float[rewards.length];
        float[] returns = new float[rewards.length];

        float gae = 0.0f;

        for (int t = rewards.length - 1; t >= 0; t--) {
            float nextValue =
                    t == rewards.length - 1
                            ? bootstrapValue
                            : values[t + 1];

            float nonTerminal = dones[t] ? 0.0f : 1.0f;

            float delta =
                    rewards[t]
                            + gamma * nextValue * nonTerminal
                            - values[t];

            gae =
                    delta
                            + gamma
                            * gaeLambda
                            * nonTerminal
                            * gae;

            advantages[t] = finite(gae);
            returns[t] = finite(advantages[t] + values[t]);
        }

        return new float[][]{
                advantages,
                returns
        };
    }

    public static double jointLogProb(
            PolicyValueNetwork.Output out,
            int move,
            int yaw,
            int pitch,
            int jump,
            int sprint,
            int sneak,
            int attack) {

        if (out == null) {
            throw new IllegalArgumentException("network output cannot be null");
        }

        return
                Categorical.logProb(out.moveLogits, move)
                + Categorical.logProb(out.yawLogits, yaw)
                + Categorical.logProb(out.pitchLogits, pitch)
                + Categorical.logProb(out.jumpLogits, jump)
                + Categorical.logProb(out.sprintLogits, sprint)
                + Categorical.logProb(out.sneakLogits, sneak)
                + Categorical.logProb(out.attackLogits, attack);
    }

    public static double jointEntropy(
            PolicyValueNetwork.Output out) {

        if (out == null) {
            throw new IllegalArgumentException("network output cannot be null");
        }

        return
                Categorical.entropy(out.moveLogits)
                + Categorical.entropy(out.yawLogits)
                + Categorical.entropy(out.pitchLogits)
                + Categorical.entropy(out.jumpLogits)
                + Categorical.entropy(out.sprintLogits)
                + Categorical.entropy(out.sneakLogits)
                + Categorical.entropy(out.attackLogits);
    }

    /**
     * Gradient of categorical policy loss plus entropy regularization
     * with respect to logits.
     */
    public static double[] policyLogitGradient(
            double[] logits,
            int action,
            double dLossDLogProb,
            double entropyCoef) {

        double[] probabilities = Categorical.softmax(logits);
        double entropy = Categorical.entropy(logits);

        double[] gradient = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            double p = probabilities[i];

            // d(log p_a)/d(logit_i) = 1[i=a] - p_i
            double dLogProbDLogit =
                    (i == action ? 1.0 : 0.0) - p;

            // dH/dz_i = p_i * (-H - log(p_i))
            double dEntropyDLogit =
                    p * (-entropy - safeLog(p));

            // Loss = policyLoss - entropyCoef * H
            gradient[i] =
                    dLossDLogProb * dLogProbDLogit
                    - entropyCoef * dEntropyDLogit;
        }

        return gradient;
    }

    public static float normalizeAdvantages(float[] advantages) {
        if (advantages == null || advantages.length == 0) {
            return 0.0f;
        }

        double mean = 0.0;

        for (float value : advantages) {
            mean += value;
        }

        mean /= advantages.length;

        double variance = 0.0;

        for (float value : advantages) {
            double delta = value - mean;
            variance += delta * delta;
        }

        variance /= advantages.length;

        double std = Math.sqrt(Math.max(0.0, variance))
                + 1.0e-8;

        for (int i = 0; i < advantages.length; i++) {
            advantages[i] =
                    (float) ((advantages[i] - mean) / std);
        }

        return (float) mean;
    }

    private static double safeLog(double p) {
        return Math.log(Math.max(p, 1.0e-12));
    }

    private static float finite(float value) {
        return Float.isFinite(value) ? value : 0.0f;
    }
}