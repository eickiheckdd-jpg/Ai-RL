package com.example.newgen6.rl;

/**
 * On-policy PPO update over a filled RolloutBuffer.
 * Genuine clipped surrogate + value loss + entropy (via policy grads).
 */
public final class PPOTrainer {
    private final PolicyNetwork net;
    private int updates;

    public float lastPolicyLoss;
    public float lastValueLoss;
    public float lastEntropy;

    public PPOTrainer(PolicyNetwork net) {
        this.net = net;
    }

    public int updates() {
        return updates;
    }

    public void update(RolloutBuffer buf, ContextBuffer scratchCtx) {
        if (buf.size < 2) return;

        float lastVal = 0f;
        if (!buf.done[buf.size - 1]) {
            buf.hydrateContext(buf.size - 1, scratchCtx);
            lastVal = net.evaluateValue(scratchCtx);
        }
        buf.computeAdvantages(lastVal, RLConstants.GAMMA, RLConstants.GAE_LAMBDA);

        for (int epoch = 0; epoch < RLConstants.PPO_EPOCHS; epoch++) {
            net.zeroGrads();
            int steps = 0;
            float polAcc = 0f, valAcc = 0f, entAcc = 0f;

            for (int start = 0; start < buf.size; start += RLConstants.PPO_MINIBATCH) {
                int end = Math.min(buf.size, start + RLConstants.PPO_MINIBATCH);
                for (int i = start; i < end; i++) {
                    buf.hydrateContext(i, scratchCtx);
                    ActionSample a = buf.actionAt(i);
                    net.accumulatePpoGrads(
                            scratchCtx,
                            a,
                            buf.advantage[i],
                            buf.returns[i],
                            buf.logProb[i],
                            RLConstants.PPO_CLIP);
                    steps++;
                    polAcc += Math.abs(buf.advantage[i]);
                    valAcc += Math.abs(buf.returns[i] - a.value);
                    entAcc += a.entropy;
                }
            }
            net.adamStep(RLConstants.LR);
            lastPolicyLoss = steps > 0 ? polAcc / steps : 0f;
            lastValueLoss = steps > 0 ? valAcc / steps : 0f;
            lastEntropy = steps > 0 ? entAcc / steps : 0f;
        }
        updates++;
    }
}
