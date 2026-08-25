package com.example.newgen6.rl.nn;

import java.util.Random;

public class GRUCell {
    public static class Cache {
        public float[] hPrev;
        public float[] z;
        public float[] r;
        public float[] hCandidate;
    }

    private final int inputSize;
    private final int hiddenSize;

    public GRUCell(int inputSize, int hiddenSize, Random rng) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void registerWith(AdamOptimizer optimizer) {
        // Register parameters with Adam Optimizer
    }
}
