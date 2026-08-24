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

    public void initTransient() {
        // No longer relying on global transient hidden state arrays
    }

    private void initWeights() {
        for (int i = 0; i < weights1.length; i++) for (int j = 0; j < weights1[0].length; j++) weights1[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weights2.length; i++) for (int j = 0; j < weights2[0].length; j++) weights2[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weights3.length; i++) for (int j = 0; j < weights3[0].length; j++) weights3[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
    }

    public float[] forward(float[] input, float[] output) {
        float[] h1 = new float[bias1.length];
        float[] h2 = new float[bias2.length];
        computeLayer(input, weights1, bias1, h1, true);
        computeLayer(h1, weights2, bias2, h2, true);
        computeLayer(h2, weights3, bias3, output, false);
        return h2; // Returns thread-safe hidden layer 2 activations for backprop
    }

    private void computeLayer(float[] in, float[][] w, float[] b, float[] out, boolean relu) {
        for (int j = 0; j < out.length; j++) {
            float sum = b[j];
            for (int i = 0; i < in.length; i++) sum += in[i] * w[i][j];
            out[j] = relu ? Math.max(0, sum) : sum;
        }
    }
}
