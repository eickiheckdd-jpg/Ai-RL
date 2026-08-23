package com.example.newgen6;
import java.util.Random;

public class ActorNetwork {
    private final int inputSize, outputSize;
    private final int hidden = 64;
    public final float[] w1, b1, w2, b2; // Changed to public

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
            out[o] = (float) Math.tanh(sum); // Bound to [-1, 1]
        }
        return out;
    }
}
