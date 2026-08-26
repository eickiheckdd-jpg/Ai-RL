package com.example.newgen6.rl;

/**
 * On-policy PPO update over a filled RolloutBuffer.
 * Supports spreading work across client ticks so FPS does not collapse to 0.
 */
public final class PPOTrainer {
    private final PolicyNetwork net;
    private int updates;

    public float lastPolicyLoss;
    public float lastValueLoss;
    public float lastEntropy;

    /** When non-null, an update is in progress across ticks. */
    private RolloutBuffer active;
    private ContextBuffer scratch;
    private int epoch;
    private int cursor;
    private boolean advantagesReady;
    private int stepsThisUpdate;
    private float polAcc, valAcc, entAcc;

    public PPOTrainer(PolicyNetwork net) {
        this.net = net;
    }

    public int updates() {
        return updates;
    }

    public boolean isTraining() {
        return active != null;
    }

    /**
     * Begin a PPO update. Does NOT run the full update — call {@link #tickWork()} each client tick.
     */
    public void beginUpdate(RolloutBuffer buf, ContextBuffer scratchCtx) {
        if (buf.size < 2) {
            buf.clear();
            return;
        }
        this.active = buf;
        this.scratch = scratchCtx;
        this.epoch = 0;
        this.cursor = 0;
        this.advantagesReady = false;
        this.stepsThisUpdate = 0;
        this.polAcc = 0f;
        this.valAcc = 0f;
        this.entAcc = 0f;
    }

    /**
     * Run a limited amount of PPO work. Safe to call every client tick.
     * @return true if the full update finished this call
     */
    public boolean tickWork() {
        if (active == null) return false;

        if (!advantagesReady) {
            float lastVal = 0f;
            if (!active.done[active.size - 1]) {
                active.hydrateContext(active.size - 1, scratch);
                lastVal = net.evaluateValue(scratch);
            }
            active.computeAdvantages(lastVal, RLConstants.GAMMA, RLConstants.GAE_LAMBDA);
            advantagesReady = true;
            net.zeroGrads();
            return false;
        }

        int budget = RLConstants.PPO_STEPS_PER_TICK;
        while (budget-- > 0 && active != null) {
            if (cursor >= active.size) {
                net.adamStep(RLConstants.LR);
                epoch++;
                cursor = 0;
                if (epoch >= RLConstants.PPO_EPOCHS) {
                    finish();
                    return true;
                }
                net.zeroGrads();
                polAcc = valAcc = entAcc = 0f;
                stepsThisUpdate = 0;
                continue;
            }

            int i = cursor++;
            active.hydrateContext(i, scratch);
            ActionSample a = active.actionAt(i);
            net.accumulatePpoGrads(
                    scratch,
                    a,
                    active.advantage[i],
                    active.returns[i],
                    active.logProb[i],
                    RLConstants.PPO_CLIP);
            stepsThisUpdate++;
            polAcc += Math.abs(active.advantage[i]);
            valAcc += Math.abs(active.returns[i] - a.value);
            entAcc += a.entropy;
        }
        return false;
    }

    private void finish() {
        if (stepsThisUpdate > 0) {
            lastPolicyLoss = polAcc / stepsThisUpdate;
            lastValueLoss = valAcc / stepsThisUpdate;
            lastEntropy = entAcc / stepsThisUpdate;
        }
        if (active != null) active.clear();
        active = null;
        scratch = null;
        updates++;
    }

    /** Legacy blocking update (avoid on client thread). */
    public void update(RolloutBuffer buf, ContextBuffer scratchCtx) {
        beginUpdate(buf, scratchCtx);
        while (active != null) {
            tickWork();
        }
    }
}