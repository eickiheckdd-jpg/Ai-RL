package com.example.newgen6.rl;

/**
 * Welford online mean / variance for observation normalization.
 * Critical for stable PPO with 229 mixed-scale features.
 */
public final class RunningMeanStd {
    private final int dim;
    private final double[] mean;
    private final double[] m2;
    private long count = 0;
    private final float epsilon = 1e-8f;

    public RunningMeanStd(int dim) {
        this.dim = dim;
        this.mean = new double[dim];
        this.m2 = new double[dim];
    }

    public void update(float[] x) {
        count++;
        for (int i = 0; i < dim; i++) {
            double delta = x[i] - mean[i];
            mean[i] += delta / count;
            double delta2 = x[i] - mean[i];
            m2[i] += delta * delta2;
        }
    }

    public void normalize(float[] x, float[] out) {
        if (count < 2) {
            System.arraycopy(x, 0, out, 0, dim);
            return;
        }
        for (int i = 0; i < dim; i++) {
            double var = m2[i] / (count - 1);
            float std = (float) Math.sqrt(var + epsilon);
            out[i] = (float) ((x[i] - mean[i]) / std);
            // soft clip to avoid extreme values early
            if (out[i] > 5f) out[i] = 5f;
            if (out[i] < -5f) out[i] = -5f;
        }
    }

    public long getCount() {
        return count;
    }
}
