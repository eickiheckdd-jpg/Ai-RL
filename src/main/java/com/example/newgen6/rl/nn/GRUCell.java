package com.example.newgen6.rl.nn;

import java.util.Random;

/**
 * Single-layer GRU cell used as the "temporal encoder" from the HUD spec:
 *
 *     200 x 229 (raw ticks)  --->  temporal encoder  --->  compact latent (h)
 *
 * Rather than literally re-flattening 200 x 229 = 45,800 floats into a dense
 * network every tick (explicitly disallowed by the spec), the GRU hidden
 * state h_t is carried forward tick-to-tick:
 *
 *     h_t = GRU(obs_t, h_{t-1})
 *
 * h_t is a compressed running summary of roughly the last ~200 ticks of
 * history (bounded by the cell's forget/update gate dynamics), which is
 * exactly the "compact recurrent state" option the spec calls out as
 * suitable for 2GB RAM / Pojav. This h_t (not just obs_t) is what feeds the
 * policy/value heads, so the temporal context genuinely affects the policy -
 * it is not decorative HUD-only data.
 *
 * Equations (standard GRU, Cho et al. 2014):
 *   z_t    = sigmoid(Wz x_t + Uz h_{t-1} + bz)         // update gate
 *   r_t    = sigmoid(Wr x_t + Ur h_{t-1} + br)         // reset gate
 *   hHat_t = tanh(Wh x_t + Uh (r_t ⊙ h_{t-1}) + bh)     // candidate state
 *   h_t    = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ hHat_t         // new hidden state
 */
public final class GRUCell {

    public final int inputSize;
    public final int hiddenSize;

    // Parameters: input->hidden (W*) and hidden->hidden (U*) weight matrices + biases.
    public final float[][] Wz, Wr, Wh; // [hidden][input]
    public final float[][] Uz, Ur, Uh; // [hidden][hidden]
    public final float[] bz, br, bh;   // [hidden]

    // Gradient accumulators, same shapes as parameters. Zeroed via zeroGrad().
    public final float[][] gWz, gWr, gWh;
    public final float[][] gUz, gUr, gUh;
    public final float[] gbz, gbr, gbh;

    /** Per-step cache needed for BPTT. One instance per timestep in a training segment. */
    public static final class StepCache {
        public float[] x, hPrev, z, r, hHat, h;
    }

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

    /**
     * Random init only - NO pretrained weights, per spec section 7.
     * Glorot/Xavier-ish uniform init: U(-limit, limit), limit = sqrt(6/(fanIn+fanOut)).
     */
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

    /** Forward one timestep. Returns new hidden state and fills cache for backward(). */
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

    /**
     * Backpropagates dL/dh_t (dhNext) through this timestep, accumulating
     * parameter gradients into the g* arrays and returning dL/dh_{t-1} so
     * the caller can continue the chain backward through time (BPTT).
     */
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
        for (int i = 0; i < hiddenSize; i++) dhHatPre[i] = dhHat[i] * (1 - c.hHat[i] * c.hHat[i]); // tanh'

        float[] rh = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) rh[i] = c.r[i] * c.hPrev[i];

        // Wh, Uh, bh gradients
        for (int i = 0; i < hiddenSize; i++) {
            float g = dhHatPre[i];
            for (int j = 0; j < inputSize; j++) gWh[i][j] += g * c.x[j];
            for (int j = 0; j < hiddenSize; j++) gUh[i][j] += g * rh[j];
            gbh[i] += g;
        }

        // d(r ⊙ hPrev) = Uh^T dhHatPre
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
            dzPre[i] = dz[i] * c.z[i] * (1 - c.z[i]);   // sigmoid'
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
