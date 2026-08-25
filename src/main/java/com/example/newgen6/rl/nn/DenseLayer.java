package com.example.newgen6.rl.nn;

import java.util.Random;

/** A plain linear layer y = W x + b, with optional tanh activation, and backprop. */
public final class DenseLayer {
    public final int inDim, outDim;
    public final boolean useTanh;
    public final float[][] W; // [out][in]
    public final float[] b;   // [out]
    public final float[][] gW;
    public final float[] gb;

    public DenseLayer(int inDim, int outDim, boolean useTanh, Random rng) {
        this.inDim = inDim; this.outDim = outDim; this.useTanh = useTanh;
        W = new float[outDim][inDim];
        b = new float[outDim];
        double limit = Math.sqrt(6.0 / (inDim + outDim));
        for (int i = 0; i < outDim; i++)
            for (int j = 0; j < inDim; j++)
                W[i][j] = (float) ((rng.nextDouble() * 2 - 1) * limit);
        gW = new float[outDim][inDim];
        gb = new float[outDim];
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
        for (int i = 0; i < outDim; i++) out[i] = useTanh ? (float) Math.tanh(pre[i]) : pre[i];
        if (cache != null) { cache.x = x; cache.preAct = pre; cache.out = out; }
        return out;
    }

    /** Backprop dL/dOut through this layer; accumulates gW/gb, returns dL/dX. */
    public float[] backward(Cache c, float[] dOut) {
        float[] dPre = new float[outDim];
        for (int i = 0; i < outDim; i++) {
            dPre[i] = useTanh ? dOut[i] * (1 - c.out[i] * c.out[i]) : dOut[i];
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