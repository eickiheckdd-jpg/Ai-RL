package com.example.newgen6;
import java.util.Random;

public class CriticNetwork {
    private final int inputSize;
    private final int hidden = 64;
    private final float[] w1, b1, w2, b2;

    public CriticNetwork(int stateSize, int actionSize) {
        this.inputSize = stateSize + actionSize;
        w1 = new float[inputSize * hidden]; b1 = new float[hidden];
        w2 = new float[hidden]; b2 = new float[1];
        init(w1); init(w2);
    }

    private void init(float[] arr) {
        Random r = new Random();
        for (int i = 0; i < arr.length; i++) arr[i] = (float)(r.nextGaussian() * 0.1);
    }

    public float evaluate(float[] state, float[] action) {
        float[] combined = new float[inputSize];
        System.arraycopy(state, 0, combined, 0, state.length);
        System.arraycopy(action, 0, combined, state.length, action.length);

        float[] h = new float[hidden];
        for (int o = 0; o < hidden; o++) {
            float sum = b1[o];
            for (int j = 0; j < inputSize; j++) sum += combined[j] * w1[j * hidden + o];
            h[o] = Math.max(0, sum);
        }
        float q = b2[0];
        for (int j = 0; j < hidden; j++) q += h[j] * w2[j];
        return q;
    }
}
