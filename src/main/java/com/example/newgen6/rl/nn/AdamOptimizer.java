package com.example.newgen6.rl.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * Adam optimizer (Kingma & Ba, 2015) implemented directly over the float[][]
 * / float[] parameter and gradient arrays used by GRUCell / DenseLayer.
 *
 * Update rule per parameter theta, gradient g, timestep t:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
 *   mHat = m_t / (1 - beta1^t)          // bias correction
 *   vHat = v_t / (1 - beta2^t)          // bias correction
 *   theta -= lr * mHat / (sqrt(vHat) + eps)
 *
 * Gradients are clipped by global norm before the step (gradient clipping),
 * a standard PPO stabilizer, applied jointly across every registered tensor
 * so direction is preserved.
 */
public final class AdamOptimizer {

    private final float lr, beta1, beta2, eps;
    private int t = 0;

    private final List<float[][]> mats = new ArrayList<>();
    private final List<float[][]> matGrads = new ArrayList<>();
    private final List<float[][]> matM = new ArrayList<>();
    private final List<float[][]> matV = new ArrayList<>();

    private final List<float[]> vecs = new ArrayList<>();
    private final List<float[]> vecGrads = new ArrayList<>();
    private final List<float[]> vecM = new ArrayList<>();
    private final List<float[]> vecV = new ArrayList<>();

    public AdamOptimizer(float lr) { this(lr, 0.9f, 0.999f, 1e-8f); }

    public AdamOptimizer(float lr, float beta1, float beta2, float eps) {
        this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    }

    public void register(float[][] param, float[][] grad) {
        mats.add(param); matGrads.add(grad);
        matM.add(new float[param.length][param[0].length]);
        matV.add(new float[param.length][param[0].length]);
    }

    public void register(float[] param, float[] grad) {
        vecs.add(param); vecGrads.add(grad);
        vecM.add(new float[param.length]);
        vecV.add(new float[param.length]);
    }

    /** Computes the global L2 norm of every registered gradient tensor. */
    public double globalGradNorm() {
        double sumSq = 0.0;
        for (float[][] g : matGrads) for (float[] row : g) for (float v : row) sumSq += (double) v * v;
        for (float[] g : vecGrads) for (float v : g) sumSq += (double) v * v;
        return Math.sqrt(sumSq);
    }

    /** Scales all gradients in-place so the global norm does not exceed maxNorm. */
    public void clipGlobalNorm(double maxNorm) {
        double norm = globalGradNorm();
        if (norm <= maxNorm || norm == 0.0) return;
        float scale = (float) (maxNorm / norm);
        for (float[][] g : matGrads) for (float[] row : g) for (int i = 0; i < row.length; i++) row[i] *= scale;
        for (float[] g : vecGrads) for (int i = 0; i < g.length; i++) g[i] *= scale;
    }

    /** Performs one Adam update step over all registered parameters using their current gradients. */
    public void step() {
        t++;
        double biasCorr1 = 1.0 - Math.pow(beta1, t);
        double biasCorr2 = 1.0 - Math.pow(beta2, t);

        for (int p = 0; p < mats.size(); p++) {
            float[][] param = mats.get(p), grad = matGrads.get(p), m = matM.get(p), v = matV.get(p);
            for (int i = 0; i < param.length; i++) {
                for (int j = 0; j < param[i].length; j++) {
                    float g = grad[i][j];
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;
                    float mHat = (float) (m[i][j] / biasCorr1);
                    float vHat = (float) (v[i][j] / biasCorr2);
                    param[i][j] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
                }
            }
        }
        for (int p = 0; p < vecs.size(); p++) {
            float[] param = vecs.get(p), grad = vecGrads.get(p), m = vecM.get(p), v = vecV.get(p);
            for (int i = 0; i < param.length; i++) {
                float g = grad[i];
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                float mHat = (float) (m[i] / biasCorr1);
                float vHat = (float) (v[i] / biasCorr2);
                param[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }
}
