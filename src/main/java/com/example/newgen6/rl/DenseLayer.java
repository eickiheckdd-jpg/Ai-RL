package com.example.newgen6.rl;

import java.util.Random;

/** Dense layer with weights, bias, Adam state. Pure Java. */
public final class DenseLayer {
    public final int in;
    public final int out;
    public final float[] w; // out * in, row-major
    public final float[] b;
    public final float[] gw;
    public final float[] gb;
    public final float[] mw;
    public final float[] vw;
    public final float[] mb;
    public final float[] vb;
    public final float[] lastIn;
    public final float[] lastOut;
    public final boolean relu;

    public DenseLayer(int in, int out, boolean relu, Random rng) {
        this.in = in;
        this.out = out;
        this.relu = relu;
        this.w = new float[out * in];
        this.b = new float[out];
        this.gw = new float[out * in];
        this.gb = new float[out];
        this.mw = new float[out * in];
        this.vw = new float[out * in];
        this.mb = new float[out];
        this.vb = new float[out];
        this.lastIn = new float[in];
        this.lastOut = new float[out];
        float scale = (float) Math.sqrt(2.0 / in);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (rng.nextGaussian() * scale);
        }
    }

    public void forward(float[] x, float[] y) {
        if (x.length != in || y.length != out) {
            throw new IllegalArgumentException("DenseLayer shape");
        }
        System.arraycopy(x, 0, lastIn, 0, in);
        for (int o = 0; o < out; o++) {
            float s = b[o];
            int base = o * in;
            for (int i = 0; i < in; i++) s += w[base + i] * x[i];
            if (relu) s = MathUtil.relu(s);
            y[o] = s;
            lastOut[o] = s;
        }
    }

    /** Backprop: dL/dy -> accumulate dW, db; write dL/dx into dx. */
    public void backward(float[] dy, float[] dx) {
        if (relu) {
            for (int o = 0; o < out; o++) {
                if (lastOut[o] <= 0f) dy[o] = 0f;
            }
        }
        if (dx != null) {
            java.util.Arrays.fill(dx, 0f);
            for (int o = 0; o < out; o++) {
                float g = dy[o];
                int base = o * in;
                for (int i = 0; i < in; i++) {
                    dx[i] += w[base + i] * g;
                }
            }
        }
        for (int o = 0; o < out; o++) {
            float g = dy[o];
            gb[o] += g;
            int base = o * in;
            for (int i = 0; i < in; i++) {
                gw[base + i] += g * lastIn[i];
            }
        }
    }

    public void zeroGrads() {
        java.util.Arrays.fill(gw, 0f);
        java.util.Arrays.fill(gb, 0f);
    }

    public void adamStep(float lr, float beta1, float beta2, float eps, int t) {
        float b1t = 1f - (float) Math.pow(beta1, t);
        float b2t = 1f - (float) Math.pow(beta2, t);
        for (int i = 0; i < w.length; i++) {
            float g = gw[i];
            mw[i] = beta1 * mw[i] + (1 - beta1) * g;
            vw[i] = beta2 * vw[i] + (1 - beta2) * g * g;
            float mhat = mw[i] / b1t;
            float vhat = vw[i] / b2t;
            w[i] -= lr * mhat / ((float) Math.sqrt(vhat) + eps);
        }
        for (int i = 0; i < b.length; i++) {
            float g = gb[i];
            mb[i] = beta1 * mb[i] + (1 - beta1) * g;
            vb[i] = beta2 * vb[i] + (1 - beta2) * g * g;
            float mhat = mb[i] / b1t;
            float vhat = vb[i] / b2t;
            b[i] -= lr * mhat / ((float) Math.sqrt(vhat) + eps);
        }
    }

    public void clipAccumulate(float[] flat, int offset) {
        // optional: export grads — not required for in-place adam
    }
}
