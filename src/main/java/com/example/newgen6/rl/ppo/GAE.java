package com.example.newgen6.rl.ppo;

public final class GAE {
    private GAE() {}

    /**
     * Standard Generalized Advantage Estimation, computed backward over one
     * trajectory segment.
     *
     * @param bootstrapValue V(s_T) for the state immediately after the last
     *                       stored transition (use 0 if that transition was
     *                       terminal).
     * @param advantagesOut  output array, same length as rewards
     * @param returnsOut     output array (value regression targets), same length as rewards
     */
    public static void compute(float[] rewards, float[] values, boolean[] dones,
                                float bootstrapValue, float gamma, float lambda,
                                float[] advantagesOut, float[] returnsOut) {
        int n = rewards.length;
        float lastGaeLam = 0f;
        for (int t = n - 1; t >= 0; t--) {
            float nextValue = (t == n - 1) ? bootstrapValue : values[t + 1];
            float nextNonTerminal = dones[t] ? 0f : 1f;
            float delta = rewards[t] + gamma * nextValue * nextNonTerminal - values[t];
            lastGaeLam = delta + gamma * lambda * nextNonTerminal * lastGaeLam;
            advantagesOut[t] = lastGaeLam;
            returnsOut[t] = advantagesOut[t] + values[t];
        }
    }
}
