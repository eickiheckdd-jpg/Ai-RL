package com.example.newgen6.rl.nn;

import java.util.Arrays;
import java.util.Random;

/**
 * Manually implemented GRU cell for the NewGen6 recurrent policy.
 *
 * Equations:
 *
 * z = sigmoid(Wz*x + Uz*hPrev + bz)
 * r = sigmoid(Wr*x + Ur*hPrev + br)
 * n = tanh(Wn*x + Un*(r*hPrev) + bn)
 * h = (1-z)*hPrev + z*n
 *
 * Gradients are accumulated internally until zeroGrad() is called.
 */
public final class GRUCell {

    public static final class Cache {
        float[] x;
        float[] hPrev;
        float[] z;
        float[] r;
        float[] n;
        float[] h;
        boolean valid;
    }

    private final int inputSize;
    private final int hiddenSize;

    private final float[][] wz;
    private final float[][] uz;
    private final float[] bz;

    private final float[][] wr;
    private final float[][] ur;
    private final float[] br;

    private final float[][] wn;
    private final float[][] un;
    private final float[] bn;

    private final float[][] gwz;
    private final float[][] guz;
    private final float[] gbz;

    private final float[][] gwr;
    private final float[][] gur;
    private final float[] gbr;

    private final float[][] gwn;
    private final float[][] gun;
    private final float[] gbn;

    public GRUCell(int inputSize, int hiddenSize, Random rng) {
        if (inputSize <= 0) throw new IllegalArgumentException("inputSize must be > 0");
        if (hiddenSize <= 0) throw new IllegalArgumentException("hiddenSize must be > 0");
        if (rng == null) throw new IllegalArgumentException("rng cannot be null");

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        wz = new float[hiddenSize][inputSize];
        uz = new float[hiddenSize][hiddenSize];
        bz = new float[hiddenSize];

        wr = new float[hiddenSize][inputSize];
        ur = new float[hiddenSize][hiddenSize];
        br = new float[hiddenSize];

        wn = new float[hiddenSize][inputSize];
        un = new float[hiddenSize][hiddenSize];
        bn = new float[hiddenSize];

        gwz = new float[hiddenSize][inputSize];
        guz = new float[hiddenSize][hiddenSize];
        gbz = new float[hiddenSize];

        gwr = new float[hiddenSize][inputSize];
        gur = new float[hiddenSize][hiddenSize];
        gbr = new float[hiddenSize];

        gwn = new float[hiddenSize][inputSize];
        gun = new float[hiddenSize][hiddenSize];
        gbn = new float[hiddenSize];

        initialize(rng);
    }

    private void initialize(Random rng) {
        float inputScale = (float) Math.sqrt(1.0 / inputSize);
        float hiddenScale = (float) Math.sqrt(1.0 / hiddenSize);

        initializeMatrix(wz, rng, inputScale);
        initializeMatrix(wr, rng, inputScale);
        initializeMatrix(wn, rng, inputScale);

        initializeMatrix(uz, rng, hiddenScale);
        initializeMatrix(ur, rng, hiddenScale);
        initializeMatrix(un, rng, hiddenScale);

        Arrays.fill(bz, 0.0f);
        Arrays.fill(br, 0.0f);
        Arrays.fill(bn, 0.0f);
    }

    private static void initializeMatrix(float[][] matrix, Random rng, float scale) {
        for (float[] row : matrix) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (float) (rng.nextGaussian() * scale);
            }
        }
    }

    public float[] forward(float[] x, float[] hPrev, Cache cache) {
        validateVector(x, inputSize, "GRU input");
        validateVector(hPrev, hiddenSize, "GRU hidden state");

        float[] z = new float[hiddenSize];
        float[] r = new float[hiddenSize];
        float[] n = new float[hiddenSize];
        float[] h = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            float az = bz[i];
            float ar = br[i];
            float an = bn[i];

            for (int j = 0; j < inputSize; j++) {
                az += wz[i][j] * x[j];
                ar += wr[i][j] * x[j];
                an += wn[i][j] * x[j];
            }

            float resetCandidate = 0.0f;
            for (int j = 0; j < hiddenSize; j++) {
                az += uz[i][j] * hPrev[j];
                ar += ur[i][j] * hPrev[j];
                resetCandidate += un[i][j] * (0.0f); // initialized below
            }

            z[i] = sigmoid(az);
            r[i] = sigmoid(ar);

            float resetHiddenContribution = 0.0f;
            for (int j = 0; j < hiddenSize; j++) {
                resetHiddenContribution += un[i][j] * (r[j] * hPrev[j]);
            }

            n[i] = (float) Math.tanh(an - resetCandidate + resetHiddenContribution);
            h[i] = (1.0f - z[i]) * hPrev[i] + z[i] * n[i];
        }

        // The candidate above must use each row's own Un term. Recompute n
        // cleanly from the exact GRU equation to avoid any stale intermediate.
        for (int i = 0; i < hiddenSize; i++) {
            float an = bn[i];

            for (int j = 0; j < inputSize; j++) {
                an += wn[i][j] * x[j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                an += un[i][j] * (r[j] * hPrev[j]);
            }

            n[i] = (float) Math.tanh(an);
            h[i] = (1.0f - z[i]) * hPrev[i] + z[i] * n[i];
        }

        if (cache != null) {
            cache.x = x.clone();
            cache.hPrev = hPrev.clone();
            cache.z = z.clone();
            cache.r = r.clone();
            cache.n = n.clone();
            cache.h = h.clone();
            cache.valid = true;
        }

        sanitize(h);
        return h;
    }

    /**
     * Backpropagates dLoss/dh and accumulates parameter gradients.
     *
     * Returns dLoss/dhPrev.
     */
    public float[] backward(
            float[] gradH,
            float[] x,
            float[] hPrev,
            Cache cache) {

        if (cache == null || !cache.valid) {
            throw new IllegalArgumentException("Valid GRU forward cache required");
        }

        validateVector(gradH, hiddenSize, "GRU hidden gradient");

        float[] gradX = new float[inputSize];
        float[] gradHPrev = new float[hiddenSize];

        float[] gradZ = new float[hiddenSize];
        float[] gradR = new float[hiddenSize];
        float[] gradN = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            gradN[i] = gradH[i] * cache.z[i];
            gradZ[i] = gradH[i] * (cache.n[i] - cache.hPrev[i]);
            gradHPrev[i] = gradH[i] * (1.0f - cache.z[i]);
        }

        float[] gradAN = new float[hiddenSize];
        float[] gradAZ = new float[hiddenSize];
        float[] gradAR = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            gradAN[i] = gradN[i] * (1.0f - cache.n[i] * cache.n[i]);
            gradAZ[i] = gradZ[i] * cache.z[i] * (1.0f - cache.z[i]);
        }

        /*
         * The candidate path contains:
         *
         * aN = Wn*x + Un*(r*hPrev) + bn
         *
         * So:
         *
         * d(r*hPrev) = Un^T * d aN
         */
        float[] gradResetHidden = new float[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            float g = gradAN[i];
            gbn[i] += g;

            for (int j = 0; j < inputSize; j++) {
                gwn[i][j] += g * cache.x[j];
                gradX[j] += g * wn[i][j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                float rh = cache.r[j] * cache.hPrev[j];
                gun[i][j] += g * rh;
                gradResetHidden[j] += g * un[i][j];
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            gradR[j] = gradResetHidden[j] * cache.hPrev[j];
            gradHPrev[j] += gradResetHidden[j] * cache.r[j];
        }

        for (int i = 0; i < hiddenSize; i++) {
            gradAR[i] = gradR[i] * cache.r[i] * (1.0f - cache.r[i]);

            float gz = gradAZ[i];
            float gr = gradAR[i];

            gbz[i] += gz;
            gbr[i] += gr;

            for (int j = 0; j < inputSize; j++) {
                gwz[i][j] += gz * cache.x[j];
                gwr[i][j] += gr * cache.x[j];

                gradX[j] += gz * wz[i][j];
                gradX[j] += gr * wr[i][j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                guz[i][j] += gz * cache.hPrev[j];
                gur[i][j] += gr * cache.hPrev[j];

                gradHPrev[j] += gz * uz[i][j];
                gradHPrev[j] += gr * ur[i][j];
            }
        }

        sanitize(gradX);
        sanitize(gradHPrev);
        sanitizeGradients();
        return gradHPrev;
    }

    public void zeroGrad() {
        clear(gwz);
        clear(guz);
        Arrays.fill(gbz, 0.0f);

        clear(gwr);
        clear(gur);
        Arrays.fill(gbr, 0.0f);

        clear(gwn);
        clear(gun);
        Arrays.fill(gbn, 0.0f);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public float[][] weightsZ() { return wz; }
    public float[][] weightsR() { return wr; }
    public float[][] weightsN() { return wn; }

    public float[][] recurrentWeightsZ() { return uz; }
    public float[][] recurrentWeightsR() { return ur; }
    public float[][] recurrentWeightsN() { return un; }

    public float[] biasZ() { return bz; }
    public float[] biasR() { return br; }
    public float[] biasN() { return bn; }

    public float[][] gradWeightsZ() { return gwz; }
    public float[][] gradWeightsR() { return gwr; }
    public float[][] gradWeightsN() { return gwn; }

    public float[][] gradRecurrentWeightsZ() { return guz; }
    public float[][] gradRecurrentWeightsR() { return gur; }
    public float[][] gradRecurrentWeightsN() { return gun; }

    public float[] gradBiasZ() { return gbz; }
    public float[] gradBiasR() { return gbr; }
    public float[] gradBiasN() { return gbn; }

    private static float sigmoid(float x) {
        if (x >= 0.0f) {
            float e = (float) Math.exp(-x);
            return 1.0f / (1.0f + e);
        }

        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }

    private static void validateVector(float[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw new IllegalArgumentException(
                    name + " size mismatch: got "
                            + (values == null ? "null" : values.length)
                            + ", expected " + expected
            );
        }

        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        name + " contains NaN/Infinity"
                );
            }
        }
    }

    private static void clear(float[][] matrix) {
        for (float[] row : matrix) {
            Arrays.fill(row, 0.0f);
        }
    }

    private static void sanitize(float[] values) {
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                values[i] = 0.0f;
            }
        }
    }

    private void sanitizeGradients() {
        sanitizeMatrix(gwz);
        sanitizeMatrix(guz);
        sanitize(gbz);
        sanitizeMatrix(gwr);
        sanitizeMatrix(gur);
        sanitize(gbr);
        sanitizeMatrix(gwn);
        sanitizeMatrix(gun);
        sanitize(gbn);
    }

    private static void sanitizeMatrix(float[][] matrix) {
        for (float[] row : matrix) {
            sanitize(row);
        }
    }
}