package com.example.newgen6.rl;

import java.io.Serializable;
import java.util.Random;

public class MLP implements Serializable {
    private static final long serialVersionUID = 1L;

    public float[][] weights1, weights2, weights3;
    public float[] bias1, bias2, bias3;
    private final Random rand = new Random();

    public MLP(int in, int h, int out) {
        weights1 = new float[in][h]; bias1 = new float[h];
        weights2 = new float[h][h];  bias2 = new float[h];
        weights3 = new float[h][out]; bias3 = new float[out];
        initWeights();
    }

    public void initTransient() {}

    private void initWeights() {
        // He Initialization scaling for ReLU activations
        float scale1 = (float) Math.sqrt(2.0 / weights1.length);
        float scale2 = (float) Math.sqrt(2.0 / weights2.length);
        float scale3 = (float) Math.sqrt(2.0 / weights3.length);

        for (int i = 0; i < weights1.length; i++) for (int j = 0; j < weights1[0].length; j++) weights1[i][j] = (float) rand.nextGaussian() * scale1;
        for (int i = 0; i < weights2.length; i++) for (int j = 0; j < weights2[0].length; j++) weights2[i][j] = (float) rand.nextGaussian() * scale2;
        for (int i = 0; i < weights3.length; i++) for (int j = 0; j < weights3[0].length; j++) weights3[i][j] = (float) rand.nextGaussian() * scale3;
    }

    // Forward pass returning layer activations for backprop
    public float[][] forwardDetailed(float[] input, float[] output) {
        float[] h1 = new float[bias1.length];
        float[] h2 = new float[bias2.length];

        computeLayer(input, weights1, bias1, h1, true);
        computeLayer(h1, weights2, bias2, h2, true);
        computeLayer(h2, weights3, bias3, output, false);

        return new float[][]{input, h1, h2, output};
    }

    public void forward(float[] input, float[] output) {
        forwardDetailed(input, output);
    }

    private void computeLayer(float[] in, float[][] w, float[] b, float[] out, boolean relu) {
        for (int j = 0; j < out.length; j++) {
            float sum = b[j];
            for (int i = 0; i < in.length; i++) sum += in[i] * w[i][j];
            out[j] = relu ? Math.max(0, sum) : sum;
        }
    }

    // Full Backpropagation across ALL layers
    public void trainBackward(float[][] activations, float[] outputGradients, float lr) {
        float[] in = activations[0];
        float[] h1 = activations[1];
        float[] h2 = activations[2];

        int inDim = weights1.length;
        int hDim = weights2.length;
        int outDim = weights3[0].length;

        float[] dH2 = new float[hDim];
        float[] dH1 = new float[hDim];

        // 1. Backpropagate Output Layer (weights3, bias3)
        for (int j = 0; j < outDim; j++) {
            float grad = clip(outputGradients[j], -5.0f, 5.0f);
            bias3[j] += lr * grad;
            for (int i = 0; i < hDim; i++) {
                weights3[i][j] += lr * grad * h2[i];
                dH2[i] += grad * weights3[i][j];
            }
        }

        // 2. Backpropagate Hidden Layer 2 (weights2, bias2) + ReLU Derivative
        for (int j = 0; j < hDim; j++) {
            if (h2[j] <= 0) continue; // ReLU derivative
            float grad = clip(dH2[j], -5.0f, 5.0f);
            bias2[j] += lr * grad;
            for (int i = 0; i < hDim; i++) {
                weights2[i][j] += lr * grad * h1[i];
                dH1[i] += grad * weights2[i][j];
            }
        }

        // 3. Backpropagate Hidden Layer 1 (weights1, bias1) + ReLU Derivative
        for (int j = 0; j < hDim; j++) {
            if (h1[j] <= 0) continue; // ReLU derivative
            float grad = clip(dH1[j], -5.0f, 5.0f);
            bias1[j] += lr * grad;
            for (int i = 0; i < inDim; i++) {
                weights1[i][j] += lr * grad * in[i];
            }
        }
    }

    private float clip(float val, float min, float max) {
        return Math.max(min, Math.min(max, val));
    }
}
