package com.example.newgen6.rl.nn;

import java.util.Random;

public final class GRUCell {

    public final int inputSize;
    public final int hiddenSize;

    public final float[][] Wz, Wr, Wh; 
    public final float[][] Uz, Ur, Uh; 
    public final float[] bz, br, bh;   

    public final float[][] gWz, gWr, gWh;
    public final float[][] gUz, gUr, gUh;
    public final float[] gbz, gbr, gbh;

    public static final class StepCache {
        public float[] x, hPrev, z, r, hHat, h;
    }

    // Added alias for compatibility with PolicyValueNetwork
    public static final class Cache extends StepCache {}

    public GRUCell(int inputSize, int hiddenSize, Random rng) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        Wz = newMatrix(hiddenSize, inputSize, rng);
        Wr = newMatrix(hiddenSize, inputSize, rng);
        Wh = newMatrix(hiddenSize, inputSize, rng);
        Uz = newMatrix(hiddenSize, hiddenSize, rng);
        Ur = newMatrix(hiddenSize, hiddenSize, rng);
        Uh = newMatrix(hiddenSize, hiddenSize, rng);
        bz = new float[hiddenSize];
        br = new float[hiddenSize];
        bh = new float[hiddenSize];

        gWz = zerosLike(Wz); gWr = zerosLike(Wr); gWh = zerosLike(Wh);
        gUz = zerosLike(Uz); gUr = zerosLike(Ur); gUh = zerosLike(Uh);
        gbz = new float[hiddenSize]; gbr = new float[hiddenSize]; gbh = new float[hiddenSize];
    }

    private static float[][] newMatrix(int rows, int cols, Random rng) {
        float[][] m = new float[rows][cols];
        double limit = Math.sqrt(6.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (float) ((rng.nextDouble() * 2 - 1) * limit);
        return m;
    }

    private static float[][] zerosLike(float[][] m) {
        return new float[m.length][m[0].length];
    }

    private static float sigmoid(float v) { return (float) (1.0 / (1.0 + Math.exp(-v))); }

    private static float[] matVecPlusMatVecPlusBias(float[][] W, float[] x, float[][] U, float[] h, float[] b) {
        int n = W.length;
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            float sum = b[i];
            float[] wi = W[i];
            for (int j = 0; j < wi.length; j++) sum += wi[j] * x[j];
            float[] ui = U[i];
            for (int j = 0; j < ui.length; j++) sum += ui[j] * h[j];
            out[i] = sum;
        }
        return out;
    }

    public float[] forward(float[] x, float[] hPrev, StepCache cache) {
        float[] zPre = matVecPlusMatVecPlusBias(Wz, x, Uz, hPrev, bz);
        float[] rPre = matVecPlusMatVecPlusBias(Wr, x, Ur, hPrev, br);
        float[] z = new float[hiddenSize];
        float[] r = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) { z[i] = sigmoid(zPre[i]); r[i] = sigmoid(rPre[i]); }

        float[] rh = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) rh[i] = r[i] * hPrev[i];

        float[] hHatPre = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            float sum = bh[i];
            float[] wi = Wh[i];
            for (int j = 0; j < inputSize; j++) sum += wi[j] * x[j];
            float[] ui = Uh[i];
            for (int j = 0; j < hiddenSize; j++) sum += ui[j] * rh[j];
            hHatPre[i] = sum;
        }
        float[] hHat = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) hHat[i] = (float) Math.tanh(hHatPre[i]);

        float[] h = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) h[i] = (1 - z[i]) * hPrev[i] + z[i] * hHat[i];

        if (cache != null) {
            cache.x = x; cache.hPrev = hPrev; cache.z = z; cache.r = r; cache.hHat = hHat; cache.h = h;
        }
        return h;
    }

    public float[] backward(StepCache c, float[] dhNext) {
        float[] dz = new float[hiddenSize];
        float[] dhHat = new float[hiddenSize];
        float[] dhPrev = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            dz[i] = dhNext[i] * (c.hHat[i] - c.hPrev[i]);
            dhHat[i] = dhNext[i] * c.z[i];
            dhPrev[i] += dhNext[i] * (1 - c.z[i]);
        }

        float[] dhHatPre = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) dhHatPre[i] = dhHat[i] * (1 - c.hHat[i] * c.hHat[i]);

        float[] rh = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) rh[i] = c.r[i] * c.hPrev[i];

        for (int i = 0; i < hiddenSize; i++) {
            float g = dhHatPre[i];
            for (int j = 0; j < inputSize; j++) gWh[i][j] += g * c.x[j];
            for (int j = 0; j < hiddenSize; j++) gUh[i][j] += g * rh[j];
            gbh[i] += g;
        }

        float[] dRh = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            float g = dhHatPre[i];
            for (int j = 0; j < hiddenSize; j++) dRh[j] += Uh[i][j] * g;
        }

        float[] dr = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dr[i] = dRh[i] * c.hPrev[i];
            dhPrev[i] += dRh[i] * c.r[i];
        }

        float[] dzPre = new float[hiddenSize];
        float[] drPre = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dzPre[i] = dz[i] * c.z[i] * (1 - c.z[i]);
            drPre[i] = dr[i] * c.r[i] * (1 - c.r[i]);
        }

        for (int i = 0; i < hiddenSize; i++) {
            float gz = dzPre[i];
            for (int j = 0; j < inputSize; j++) gWz[i][j] += gz * c.x[j];
            for (int j = 0; j < hiddenSize; j++) gUz[i][j] += gz * c.hPrev[j];
            gbz[i] += gz;

            float gr = drPre[i];
            for (int j = 0; j < inputSize; j++) gWr[i][j] += gr * c.x[j];
            for (int j = 0; j < hiddenSize; j++) gUr[i][j] += gr * c.hPrev[j];
            gbr[i] += gr;
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                dhPrev[j] += Uz[i][j] * dzPre[i];
                dhPrev[j] += Ur[i][j] * drPre[i];
            }
        }

        return dhPrev;
    }

    public void zeroGrad() {
        zero(gWz); zero(gWr); zero(gWh);
        zero(gUz); zero(gUr); zero(gUh);
        java.util.Arrays.fill(gbz, 0f); java.util.Arrays.fill(gbr, 0f); java.util.Arrays.fill(gbh, 0f);
    }

    private static void zero(float[][] m) { for (float[] row : m) java.util.Arrays.fill(row, 0f); }
}
