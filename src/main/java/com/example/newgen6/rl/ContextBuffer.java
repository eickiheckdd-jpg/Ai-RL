package com.example.newgen6.rl;

/**
 * Ring buffer of the last CONTEXT_TICKS observations (200 × 229).
 * Primitive storage, no per-tick allocation after construction.
 */
public final class ContextBuffer {
    private final float[] data; // [tick * OBS + feature]
    private int write;
    private int count;

    public ContextBuffer() {
        this.data = new float[RLConstants.CONTEXT_TICKS * RLConstants.OBSERVATION_SIZE];
        this.write = 0;
        this.count = 0;
    }

    public void clear() {
        java.util.Arrays.fill(data, 0f);
        write = 0;
        count = 0;
    }

    public void push(float[] obs) {
        RLConstants.assertObsSize(obs.length);
        int base = write * RLConstants.OBSERVATION_SIZE;
        System.arraycopy(obs, 0, data, base, RLConstants.OBSERVATION_SIZE);
        write = (write + 1) % RLConstants.CONTEXT_TICKS;
        if (count < RLConstants.CONTEXT_TICKS) count++;
    }

    public int size() {
        return count;
    }

    public boolean isFull() {
        return count >= RLConstants.CONTEXT_TICKS;
    }

    /**
     * Copy observation from age ticks ago (0 = most recent).
     */
    public void copyAge(int age, float[] out) {
        RLConstants.assertObsSize(out.length);
        if (count == 0) {
            java.util.Arrays.fill(out, 0f);
            return;
        }
        int a = MathUtil.clamp(age, 0, count - 1);
        int idx = write - 1 - a;
        while (idx < 0) idx += RLConstants.CONTEXT_TICKS;
        System.arraycopy(data, idx * RLConstants.OBSERVATION_SIZE, out, 0, RLConstants.OBSERVATION_SIZE);
    }

    /** Mean over available history into out[0..OBS). */
    public void mean(float[] out) {
        RLConstants.assertObsSize(out.length);
        java.util.Arrays.fill(out, 0f);
        if (count == 0) return;
        for (int age = 0; age < count; age++) {
            int idx = write - 1 - age;
            while (idx < 0) idx += RLConstants.CONTEXT_TICKS;
            int base = idx * RLConstants.OBSERVATION_SIZE;
            for (int f = 0; f < RLConstants.OBSERVATION_SIZE; f++) {
                out[f] += data[base + f];
            }
        }
        float inv = 1f / count;
        for (int f = 0; f < RLConstants.OBSERVATION_SIZE; f++) out[f] *= inv;
    }

    public float featureAtAge(int age, int feature) {
        if (count == 0) return 0f;
        int a = MathUtil.clamp(age, 0, count - 1);
        int idx = write - 1 - a;
        while (idx < 0) idx += RLConstants.CONTEXT_TICKS;
        return data[idx * RLConstants.OBSERVATION_SIZE + feature];
    }
}
