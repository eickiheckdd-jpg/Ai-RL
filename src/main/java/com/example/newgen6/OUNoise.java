package com.example.newgen6;
import java.util.Random;

public class OUNoise {
    private final int size;
    private final float theta = 0.15f;
    private final float sigma = 0.2f;
    private final float[] state;
    private final Random random = new Random();

    public OUNoise(int size) {
        this.size = size;
        this.state = new float[size];
    }

    public float[] sample() {
        float[] noise = new float[size];
        for (int i = 0; i < size; i++) {
            float dx = theta * (-state[i]) + sigma * (float) random.nextGaussian();
            state[i] += dx;
            noise[i] = state[i];
        }
        return noise;
    }
}
