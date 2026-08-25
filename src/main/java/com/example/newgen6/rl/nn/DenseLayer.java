package com.example.newgen6.rl.nn;

import java.util.Random;

public class DenseLayer {
    public enum Activation {
        NONE, RELU, TANH
    }

    public static class Cache {
        public float[] input;
        public float[] output;
    }

    private final int inSize;
    private final int outSize;
    private final Activation activation;
    public final float[][] weights;
    public final float[] biases;

    public DenseLayer(int inSize, int outSize, Activation activation, Random rng) {
        this.inSize = inSize;
        this.outSize = outSize;
        this.activation = activation;
        this.weights = new float[outSize][inSize];
        this.biases = new float[outSize];
        
        // Xavier/He initialization
        float scale = (float) Math.sqrt(2.0 / inSize);
        for (int i = 0; i < outSize; i++) {
            for (int j = 0; j < inSize; j++) {
                weights[i][j] = (float) rng.nextGaussian() * scale;
            }
        }
    }

    public void registerWith(AdamOptimizer optimizer) {
        optimizer.register(weights);
        optimizer.register(biases);
    }
}
