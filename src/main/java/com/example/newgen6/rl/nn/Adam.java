package com.example.newgen6.rl.nn;

/**
 * Adam optimizer applied uniformly across a set of Dense layers with a
 * single shared timestep, so bias-correction stays consistent across every
 * parameter tensor updated together in one PPO minibatch step.
 */
public class Adam {

    public float lr;
    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float EPS = 1e-8f;
    private int t = 0;

    public Adam(float lr) {
        this.lr = lr;
    }

    public void step(Dense... layers) {
        t++;
        float bc1 = 1f - (float) Math.pow(BETA1, t);
        float bc2 = 1f - (float) Math.pow(BETA2, t);
        for (Dense layer : layers) {
            updateLayer(layer, bc1, bc2);
        }
    }

    private void updateLayer(Dense layer, float bc1, float bc2) {
        float[][] w = layer.weights();
        float[][] gw = layer.gradWeights();
        float[][] mw = layer.mWeights();
        float[][] vw = layer.vWeights();

        for (int o = 0; o < w.length; o++) {
            for (int i = 0; i < w[o].length; i++) {
                float g = gw[o][i];
                mw[o][i] = BETA1 * mw[o][i] + (1 - BETA1) * g;
                vw[o][i] = BETA2 * vw[o][i] + (1 - BETA2) * g * g;
                float mHat = mw[o][i] / bc1;
                float vHat = vw[o][i] / bc2;
                w[o][i] -= lr * mHat / ((float) Math.sqrt(vHat) + EPS);
            }
        }

        float[] b = layer.bias();
        float[] gb = layer.gradBias();
        float[] mb = layer.mBias();
        float[] vb = layer.vBias();
        for (int o = 0; o < b.length; o++) {
            float g = gb[o];
            mb[o] = BETA1 * mb[o] + (1 - BETA1) * g;
            vb[o] = BETA2 * vb[o] + (1 - BETA2) * g * g;
            float mHat = mb[o] / bc1;
            float vHat = vb[o] / bc2;
            b[o] -= lr * mHat / ((float) Math.sqrt(vHat) + EPS);
        }
    }
}
