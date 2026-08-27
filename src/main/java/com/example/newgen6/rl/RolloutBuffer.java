package com.example.newgen6.rl;

/**
 * Fixed-size circular rollout buffer. Extremely memory-conscious.
 * Stores only the last ROLLOUT_STEPS transitions.
 */
public final class RolloutBuffer {
    private final int capacity;
    private final float[][] obs;
    private final int[] actions;
    private final float[] rewards;
    private final float[] values;
    private final float[] logProbs;
    private final float[][] looks;      // continuous look targets
    private final boolean[] dones;

    private int pos = 0;
    private int size = 0;

    public RolloutBuffer(int capacity, int obsDim) {
        this.capacity = capacity;
        obs = new float[capacity][obsDim];
        actions = new int[capacity];
        rewards = new float[capacity];
        values = new float[capacity];
        logProbs = new float[capacity];
        looks = new float[capacity][2];
        dones = new boolean[capacity];
    }

    public void add(float[] o, int a, float r, float v, float lp, float[] look, boolean done) {
        System.arraycopy(o, 0, obs[pos], 0, o.length);
        actions[pos] = a;
        rewards[pos] = r;
        values[pos] = v;
        logProbs[pos] = lp;
        looks[pos][0] = look[0];
        looks[pos][1] = look[1];
        dones[pos] = done;
        pos = (pos + 1) % capacity;
        if (size < capacity) size++;
    }

    public void clear() {
        pos = 0;
        size = 0;
    }

    public int size() { return size; }

    public float[][] getObs() { return obs; }
    public int[] getActions() { return actions; }
    public float[] getRewards() { return rewards; }
    public float[] getValues() { return values; }
    public float[] getLogProbs() { return logProbs; }
    public float[][] getLooks() { return looks; }
    public boolean[] getDones() { return dones; }

    /** Compute GAE advantages in-place (returns advantages + returns) */
    public void computeGAE(float lastValue, float gamma, float lambda, float[] advantages, float[] returns) {
        float gae = 0f;
        for (int t = size - 1; t >= 0; t--) {
            float nextVal = (t == size - 1) ? lastValue : values[t + 1];
            float nextNonTerminal = dones[t] ? 0f : 1f;
            float delta = rewards[t] + gamma * nextVal * nextNonTerminal - values[t];
            gae = delta + gamma * lambda * nextNonTerminal * gae;
            advantages[t] = gae;
            returns[t] = gae + values[t];
        }
    }
}
