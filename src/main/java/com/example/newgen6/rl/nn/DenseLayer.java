package com.example.newgen6.rl.nn;

import java.util.Random;

public final class DenseLayer {

    public enum Activation {
        NONE, RELU, TANH
    }

    public final int inDim, outDim;
    public final Activation activation;
    public final float[][] W; // [out][in]
    public final float[] b;   // [out]
    public final float[][] gW;
    public final float[] gb;

    public DenseLayer(int inDim, int outDim, Activation activation, Random rng) {
        this.inDim = inDim; this.outDim = outDim; this.activation = activation;
        W = new float[outDim][inDim];
        b = new float[outDim];
        double limit = Math.sqrt(6.0 / (inDim + outDim));
        for (int i = 0; i < outDim; i++)
            for (int j = 0; j < inDim; j++)
                W[i][j] = (float) ((rng.nextDouble() * 2 - 1) * limit);
        gW = new float[outDim][inDim];
        gb = new float[outDim];
    }

    // Backward compatibility constructor for boolean useTanh
    public DenseLayer(int inDim, int outDim, boolean useTanh, Random rng) {
        this(inDim, outDim, useTanh ? Activation.TANH : Activation.NONE, rng);
    }

    public static final class Cache { public float[] x, preAct, out; }

    public float[] forward(float[] x, Cache cache) {
        float[] pre = new float[outDim];
        for (int i = 0; i < outDim; i++) {
            float sum = b[i];
            float[] wi = W[i];
            for (int j = 0; j < inDim; j++) sum += wi[j] * x[j];
            pre[i] = sum;
        }
        float[] out = new float[outDim];
        for (int i = 0; i < outDim; i++) {
            if (activation == Activation.TANH) {
                out[i] = (float) Math.tanh(pre[i]);
            } else if (activation == Activation.RELU) {
                out[i] = Math.max(0f, pre[i]);
            } else {
                out[i] = pre[i];
            }
        }
        if (cache != null) { cache.x = x; cache.preAct = pre; cache.out = out; }
        return out;
    }

    public float[] backward(Cache c, float[] dOut) {
        float[] dPre = new float[outDim];
        for (int i = 0; i < outDim; i++) {
            if (activation == Activation.TANH) {
                dPre[i] = dOut[i] * (1 - c.out[i] * c.out[i]);
            } else if (activation == Activation.RELU) {
                dPre[i] = c.out[i] > 0 ? dOut[i] : 0f;
            } else {
                dPre[i] = dOut[i];
            }
        }
        float[] dX = new float[inDim];
        for (int i = 0; i < outDim; i++) {
            float g = dPre[i];
            float[] wi = W[i];
            for (int j = 0; j < inDim; j++) {
                gW[i][j] += g * c.x[j];
                dX[j] += wi[j] * g;
            }
            gb[i] += g;
        }
        return dX;
    }

    public void zeroGrad() {
        for (float[] row : gW) java.util.Arrays.fill(row, 0f);
        java.util.Arrays.fill(gb, 0f);
    }
}
