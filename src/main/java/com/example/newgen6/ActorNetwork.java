package com.example.newgen6;

import java.util.Random;

public class ActorNetwork {
    public final int inputSize, outputSize;
    public final int hidden = 64;
    public final float[] w1, b1, w2, b2;

    public ActorNetwork(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        w1 = new float[inputSize * hidden]; b1 = new float[hidden];
        w2 = new float[hidden * outputSize]; b2 = new float[outputSize];
        init(w1); init(w2);
    }

    private void init(float[] arr) {
        Random r = new Random();
        for (int i = 0; i < arr.length; i++) arr[i] = (float)(r.nextGaussian() * 0.1);
    }

    public float[] forward(float[] input, float[] out) {
        float[] h = new float[hidden];
        for (int o = 0; o < hidden; o++) {
            float sum = b1[o];
            for (int j = 0; j < inputSize; j++) sum += input[j] * w1[j * hidden + o];
            h[o] = Math.max(0, sum); // ReLU
        }
        for (int o = 0; o < outputSize; o++) {
            float sum = b2[o];
            for (int j = 0; j < hidden; j++) sum += h[j] * w2[j * outputSize + o];
            out[o] = (float) Math.tanh(sum);
        }
        return out;
    }

    // Pure-Java Backpropagation for Actor
    public void train(float[] state, float[] action, float[] actionGrad, float lr) {
        float[] z1 = new float[hidden];
        float[] h = new float[hidden];
        for (int o = 0; o < hidden; o++) {
            float sum = b1[o];
            for (int j = 0; j < inputSize; j++) sum += state[j] * w1[j * hidden + o];
            z1[o] = sum;
            h[o] = Math.max(0, sum);
        }

        float[] dz2 = new float[outputSize];
        for (int o = 0; o < outputSize; o++) {
            // Policy Gradient: dLoss/dAction = -actionGrad, dAction/dZ2 = 1 - tanh^2
            dz2[o] = -actionGrad[o] * (1.0f - action[o] * action[o]);
        }

        // Update w2 and b2
        for (int o = 0; o < outputSize; o++) {
            b2[o] -= lr * dz2[o];
            for (int j = 0; j < hidden; j++) {
                w2[j * outputSize + o] -= lr * dz2[o] * h[j];
            }
        }

        // Backpropagate to hidden layer
        float[] dz1 = new float[hidden];
        for (int j = 0; j < hidden; j++) {
            float sum = 0.0f;
            for (int o = 0; o < outputSize; o++) {
                sum += dz2[o] * w2[j * outputSize + o];
            }
            dz1[j] = sum * (z1[j] > 0 ? 1.0f : 0.0f);
        }

        // Update w1 and b1
        for (int o = 0; o < hidden; o++) {
            b1[o] -= lr * dz1[o];
            for (int j = 0; j < inputSize; j++) {
                w1[j * hidden + o] -= lr * dz1[o] * state[j];
            }
        }
    }
}
