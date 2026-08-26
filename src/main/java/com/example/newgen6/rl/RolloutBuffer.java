package com.example.newgen6.rl;

/**
 * Compact rollout storage for PPO. Stores observation snapshots + actions.
 * Context is reconstructed by replaying obs into a temporary ContextBuffer during update
 * OR we store enough recent obs history per step (CONTEXT_TICKS) — too heavy.
 *
 * Practical approach: store the 229-obs per step and a short temporal window
 * (TEMPORAL_FRAMES + mean proxy via running mean) for on-policy updates.
 *
 * For correctness of the 200T mean feature, each step also stores the history mean.
 */
public final class RolloutBuffer {
    public final int capacity;
    public int size;

    public final float[][] obs;          // [T][229]
    public final float[][] meanCtx;      // [T][229] history mean at step
    public final float[][] recent;       // [T][TEMPORAL_FRAMES * 229] flattened ages 0..F-1

    public final int[] move;
    public final boolean[] jump, sprint, attack, sneak;
    public final int[] yawBucket, pitchBucket;
    public final float[] logProb;
    public final float[] value;
    public final float[] reward;
    public final boolean[] done;

    public final float[] advantage;
    public final float[] returns;

    public RolloutBuffer(int capacity) {
        this.capacity = capacity;
        this.obs = new float[capacity][RLConstants.OBSERVATION_SIZE];
        this.meanCtx = new float[capacity][RLConstants.OBSERVATION_SIZE];
        this.recent = new float[capacity][RLConstants.TEMPORAL_FRAMES * RLConstants.OBSERVATION_SIZE];
        this.move = new int[capacity];
        this.jump = new boolean[capacity];
        this.sprint = new boolean[capacity];
        this.attack = new boolean[capacity];
        this.sneak = new boolean[capacity];
        this.yawBucket = new int[capacity];
        this.pitchBucket = new int[capacity];
        this.logProb = new float[capacity];
        this.value = new float[capacity];
        this.reward = new float[capacity];
        this.done = new boolean[capacity];
        this.advantage = new float[capacity];
        this.returns = new float[capacity];
    }

    public void clear() {
        size = 0;
    }

    public boolean isFull() {
        return size >= capacity;
    }

    public void add(float[] observation, ContextBuffer ctx, ActionSample a, float r, boolean terminal) {
        if (size >= capacity) return;
        int i = size;
        System.arraycopy(observation, 0, obs[i], 0, RLConstants.OBSERVATION_SIZE);
        ctx.mean(meanCtx[i]);
        for (int age = 0; age < RLConstants.TEMPORAL_FRAMES; age++) {
            ctx.copyAge(age, obsScratch);
            System.arraycopy(obsScratch, 0, recent[i], age * RLConstants.OBSERVATION_SIZE, RLConstants.OBSERVATION_SIZE);
        }
        move[i] = a.move;
        jump[i] = a.jump;
        sprint[i] = a.sprint;
        attack[i] = a.attack;
        sneak[i] = a.sneak;
        yawBucket[i] = a.yawBucket;
        pitchBucket[i] = a.pitchBucket;
        logProb[i] = a.logProb;
        value[i] = a.value;
        reward[i] = r;
        done[i] = terminal;
        size++;
    }

    private final float[] obsScratch = new float[RLConstants.OBSERVATION_SIZE];

    /** GAE(λ). lastValue = bootstrap if not done. */
    public void computeAdvantages(float lastValue, float gamma, float lambda) {
        float gae = 0f;
        for (int t = size - 1; t >= 0; t--) {
            float nextVal = (t == size - 1) ? lastValue : value[t + 1];
            float mask = done[t] ? 0f : 1f;
            float delta = reward[t] + gamma * nextVal * mask - value[t];
            gae = delta + gamma * lambda * mask * gae;
            advantage[t] = gae;
            returns[t] = advantage[t] + value[t];
        }
        // normalize advantages
        float mean = 0f;
        for (int t = 0; t < size; t++) mean += advantage[t];
        mean /= Math.max(1, size);
        float var = 0f;
        for (int t = 0; t < size; t++) {
            float d = advantage[t] - mean;
            var += d * d;
        }
        float std = (float) Math.sqrt(var / Math.max(1, size)) + 1e-8f;
        for (int t = 0; t < size; t++) advantage[t] = (advantage[t] - mean) / std;
    }

    /** Rebuild a ContextBuffer-like view for step i into target ctx (clears first). */
    public void hydrateContext(int i, ContextBuffer ctx) {
        ctx.clear();
        // push oldest of the short window first so age-0 is newest
        for (int age = RLConstants.TEMPORAL_FRAMES - 1; age >= 0; age--) {
            System.arraycopy(recent[i], age * RLConstants.OBSERVATION_SIZE, obsScratch, 0, RLConstants.OBSERVATION_SIZE);
            ctx.push(obsScratch);
        }
        // inject mean via extra pushes of meanCtx so mean() approximates stored mean
        // (lightweight approximation for on-policy update)
        for (int k = 0; k < 8; k++) ctx.push(meanCtx[i]);
    }

    public ActionSample actionAt(int i) {
        ActionSample a = new ActionSample();
        a.move = move[i];
        a.jump = jump[i];
        a.sprint = sprint[i];
        a.attack = attack[i];
        a.sneak = sneak[i];
        a.yawBucket = yawBucket[i];
        a.pitchBucket = pitchBucket[i];
        a.logProb = logProb[i];
        a.value = value[i];
        a.applyBuckets();
        return a;
    }
}
