package com.example.newgen6.rl;

import java.io.Serializable;
import java.util.Random;

public class MLP implements Serializable {
    private static final long serialVersionUID = 1L;

    public float[][] weights1, weights2, weights3;
    public float[] bias1, bias2, bias3;
    public transient float[] hidden1, hidden2;
    private final Random rand = new Random();

    public MLP(int in, int h, int out) {
        weights1 = new float[in][h]; bias1 = new float[h];
        weights2 = new float[h][h];  bias2 = new float[h];
        weights3 = new float[h][out]; bias3 = new float[out];
        hidden1 = new float[h]; hidden2 = new float[h];
        initWeights();
    }

    public void initTransient() {
        if (hidden1 == null) hidden1 = new float[bias1.length];
        if (hidden2 == null) hidden2 = new float[bias2.length];
    }

    private void initWeights() {
        for (int i = 0; i < weights1.length; i++) for (int j = 0; j < weights1[0].length; j++) weights1[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weights2.length; i++) for (int j = 0; j < weights2[0].length; j++) weights2[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
        for (int i = 0; i < weights3.length; i++) for (int j = 0; j < weights3[0].length; j++) weights3[i][j] = (rand.nextFloat() - 0.5f) * 0.1f;
    }

    public void forward(float[] input, float[] output) {
        initTransient();
        computeLayer(input, weights1, bias1, hidden1, true);
        computeLayer(hidden1, weights2, bias2, hidden2, true);
        computeLayer(hidden2, weights3, bias3, output, false);
    }

    private void computeLayer(float[] in, float[][] w, float[] b, float[] out, boolean relu) {
        for (int j = 0; j < out.length; j++) {
            float sum = b[j];
            for (int i = 0; i < in.length; i++) sum += in[i] * w[i][j];
            out[j] = relu ? Math.max(0, sum) : sum;
        }
    }
}
