package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.nn.Categorical;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

public final class PPOMath {

    private PPOMath() {}

    /**
     * Computes Generalized Advantage Estimation (GAE) and Returns with episode boundary masking.
     */
    public static float[][] computeGAE(
            float[] rewards, 
            float[] values, 
            boolean[] dones, 
            float bootstrapValue, 
            float gamma, 
            float gaeLambda) {

        int n = rewards.length;
        float[] advantages = new float[n];
        float[] returns = new float[n];

        float gae = 0.0f;
        for (int t = n - 1; t >= 0; t--) {
            float nextValue = (t == n - 1) ? bootstrapValue : values[t + 1];
            float nonTerminal = dones[t] ? 0.0f : 1.0f;

            // delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            float delta = rewards[t] + gamma * nextValue * nonTerminal - values[t];
            // A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = delta + gamma * gaeLambda * nonTerminal * gae;

            advantages[t] = gae;
            returns[t] = gae + values[t];
        }

        return new float[][]{ advantages, returns };
    }

    public static double jointLogProb(
            PolicyValueNetwork.Output out,
            int move, int yaw, int pitch, int jump, int sprint, int sneak, int attack) {

        return Categorical.logProb(out.moveLogits, move)
             + Categorical.logProb(out.yawLogits, yaw)
             + Categorical.logProb(out.pitchLogits, pitch)
             + Categorical.logProb(out.jumpLogits, jump)
             + Categorical.logProb(out.sprintLogits, sprint)
             + Categorical.logProb(out.sneakLogits, sneak)
             + Categorical.logProb(out.attackLogits, attack);
    }

    public static double jointEntropy(PolicyValueNetwork.Output out) {
        return Categorical.entropy(out.moveLogits)
             + Categorical.entropy(out.yawLogits)
             + Categorical.entropy(out.pitchLogits)
             + Categorical.entropy(out.jumpLogits)
             + Categorical.entropy(out.sprintLogits)
             + Categorical.entropy(out.sneakLogits)
             + Categorical.entropy(out.attackLogits);
    }
}
