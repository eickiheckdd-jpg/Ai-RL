package com.example.newgen6.rl;

final class MathUtil {
    private MathUtil() {}

    static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /** Integer clamp — required so AimBuckets/ContextBuffer do not get float→int lossy conversion. */
    static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static float tanh(float x) {
        if (x > 20f) return 1f;
        if (x < -20f) return -1f;
        float e = (float) Math.exp(2.0 * x);
        return (e - 1f) / (e + 1f);
    }

    static float sigmoid(float x) {
        if (x > 20f) return 1f;
        if (x < -20f) return 0f;
        return 1f / (1f + (float) Math.exp(-x));
    }

    static float relu(float x) {
        return x > 0f ? x : 0f;
    }

    /** Shortest signed yaw delta in degrees, range (-180, 180]. */
    static float yawDeltaDeg(float fromYaw, float toYaw) {
        float d = (toYaw - fromYaw) % 360f;
        if (d > 180f) d -= 360f;
        if (d <= -180f) d += 360f;
        return d;
    }

    static float wrapYaw(float yaw) {
        float y = yaw % 360f;
        if (y >= 180f) y -= 360f;
        if (y < -180f) y += 360f;
        return y;
    }

    static void softmaxInPlace(float[] logits, int off, int n) {
        float max = logits[off];
        for (int i = 1; i < n; i++) {
            float v = logits[off + i];
            if (v > max) max = v;
        }
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            float e = (float) Math.exp(logits[off + i] - max);
            logits[off + i] = e;
            sum += e;
        }
        float inv = sum > 0f ? 1f / sum : 0f;
        for (int i = 0; i < n; i++) {
            logits[off + i] *= inv;
        }
    }

    static int sampleCategorical(float[] probs, int off, int n, java.util.Random rng) {
        float r = rng.nextFloat();
        float c = 0f;
        for (int i = 0; i < n; i++) {
            c += probs[off + i];
            if (r <= c) return i;
        }
        return n - 1;
    }

    static float logProbCategorical(float[] probs, int off, int index) {
        float p = Math.max(probs[off + index], 1e-8f);
        return (float) Math.log(p);
    }

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static void axpy(float[] y, float[] x, float a, int n) {
        for (int i = 0; i < n; i++) y[i] += a * x[i];
    }

    static float l2Norm(float[] g) {
        double s = 0;
        for (float v : g) s += (double) v * v;
        return (float) Math.sqrt(s);
    }

    static void scale(float[] a, float s) {
        for (int i = 0; i < a.length; i++) a[i] *= s;
    }

    static boolean finite(float v) {
        return !Float.isNaN(v) && !Float.isInfinite(v);
    }

    static float sane(float v) {
        return finite(v) ? v : 0f;
    }
}