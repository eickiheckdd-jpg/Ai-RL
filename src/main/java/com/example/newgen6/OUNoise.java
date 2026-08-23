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
            // OU process differential update
            float dx = theta * (-state[i]) + sigma * (float) random.nextGaussian();
            state[i] += dx;
            
            // Clamp internal noise state to prevent unbounded drift
            state[i] = Math.max(-0.5f, Math.min(0.5f, state[i]));

            // Scale noise based on channel type
            if (i < 2) {
                // Yaw and Pitch: Allow full exploration noise
                noise[i] = state[i];
            } else {
                // Movement/Attack Keys: Dampen noise 
                noise[i] = state[i] * 0.3f;
            }
        }
        return noise;
    }

    public void reset() {
        for (int i = 0; i < size; i++) {
            state[i] = 0.0f;
        }
    }
}
