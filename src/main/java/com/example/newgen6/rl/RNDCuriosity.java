package com.example.newgen6.rl;

import java.util.Random;

/**
 * Tiny Random Network Distillation curiosity module.
 * Fits easily in 2 GB / Pojav.
 *
 * Fixed random target net + small predictor.
 * Intrinsic reward = prediction error on normalized features.
 */
public final class RNDCuriosity {

    private final int inDim;
    private final int featDim;
    // target (fixed)
    private final float[][] wT;
    private final float[] bT;
    // predictor (learned, simple SGD)
    private final float[][] wP;
    private final float[] bP;

    private final float[] targetOut;
    private final float[] predOut;
    private final float[] featBuf;

    private final RunningMeanStd errNorm;
    private final Random rng = new Random(12345);

    public RNDCuriosity(int obsDim) {
        this.inDim = obsDim;
        this.featDim = Config.RND_FEAT_DIM;
        wT = randMat(featDim, inDim, 0.1f);
        bT = new float[featDim];
        wP = randMat(featDim, inDim, 0.1f);
        bP = new float[featDim];
        targetOut = new float[featDim];
        predOut = new float[featDim];
        featBuf = new float[featDim];
        errNorm = new RunningMeanStd(1);
    }

    /** Returns intrinsic reward in roughly [0, 1] after normalization */
    public float intrinsic(float[] normObs, boolean trainPredictor) {
        // target
        matVec(wT, normObs, bT, targetOut);
        // predictor
        matVec(wP, normObs, bP, predOut);

        float err = 0f;
        for (int i = 0; i < featDim; i++) {
            float d = targetOut[i] - predOut[i];
            err += d * d;
        }
        err /= featDim;

        float[] e = new float[]{err};
        errNorm.update(e);
        float[] n = new float[1];
        errNorm.normalize(e, n);
        float bonus = clamp(n[0] * 0.5f + 0.5f, 0f, 1.5f); // soft scale

        if (trainPredictor) {
            // one step SGD toward target
            float lr = 1e-3f;
            for (int i = 0; i < featDim; i++) {
                float d = predOut[i] - targetOut[i];
                bP[i] -= lr * d;
                for (int j = 0; j < inDim; j++) {
                    wP[i][j] -= lr * d * normObs[j];
                }
            }
        }
        return bonus;
    }

    private float[][] randMat(int r, int c, float s) {
        float[][] m = new float[r][c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                m[i][j] = (rng.nextFloat() * 2f - 1f) * s;
        return m;
    }

    private static void matVec(float[][] w, float[] x, float[] b, float[] out) {
        for (int i = 0; i < w.length; i++) {
            float s = b[i];
            float[] row = w[i];
            for (int j = 0; j < x.length && j < row.length; j++) s += row[j] * x[j];
            out[i] = (float) Math.tanh(s);
        }
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}