package com.example.newgen6.rl;

import java.util.Random;

public class PolicyNetwork {
    public static final int INPUT_SIZE = 45800; // 200 Ticks x 229 Features
    public static final int TRUNK_SIZE = 128;   // Latent Trunk dimension

    public static final int MOVE_BINS = 9;
    public static final int TOGGLE_BINS = 4;   // Jump, Sprint, Attack, Sneak
    public static final int YAW_BINS = 19;
    public static final int PITCH_BINS = 17;

    private final float[][] wTrunk = new float[INPUT_SIZE][TRUNK_SIZE];
    private final float[] bTrunk = new float[TRUNK_SIZE];
    private final float[] trunkActivation = new float[TRUNK_SIZE];

    private final float[][] wMove = new float[TRUNK_SIZE][MOVE_BINS];
    private final float[][] wToggle = new float[TRUNK_SIZE][TOGGLE_BINS];
    private final float[][] wYaw = new float[TRUNK_SIZE][YAW_BINS];
    private final float[][] wPitch = new float[TRUNK_SIZE][PITCH_BINS];

    public final float[] moveProbs = new float[MOVE_BINS];
    public final float[] toggleProbs = new float[TOGGLE_BINS];
    public final float[] yawProbs = new float[YAW_BINS];
    public final float[] pitchProbs = new float[PITCH_BINS];

    private final Random random = new Random();

    public PolicyNetwork() {
        initializeWeights(wTrunk, INPUT_SIZE, TRUNK_SIZE);
        initializeWeights(wMove, TRUNK_SIZE, MOVE_BINS);
        initializeWeights(wToggle, TRUNK_SIZE, TOGGLE_BINS);
        initializeWeights(wYaw, TRUNK_SIZE, YAW_BINS);
        initializeWeights(wPitch, TRUNK_SIZE, PITCH_BINS);
    }

    private void initializeWeights(float[][] matrix, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                matrix[i][j] = (random.nextFloat() * 2.0f - 1.0f) * limit;
            }
        }
    }

    public void forward(float[] flattenedContext) {
        for (int j = 0; j < TRUNK_SIZE; j++) {
            float sum = bTrunk[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += flattenedContext[i] * wTrunk[i][j];
            }
            trunkActivation[j] = Math.max(0.0f, sum);
        }

        computeSoftmax(trunkActivation, wMove, moveProbs);
        computeSigmoid(trunkActivation, wToggle, toggleProbs);
        computeSoftmax(trunkActivation, wYaw, yawProbs);
        computeSoftmax(trunkActivation, wPitch, pitchProbs);
    }

    private void computeSoftmax(float[] hidden, float[][] weights, float[] outputProbs) {
        float max = -1e9f;
        for (int j = 0; j < outputProbs.length; j++) {
            float sum = 0.0f;
            for (int i = 0; i < hidden.length; i++) {
                sum += hidden[i] * weights[i][j];
            }
            outputProbs[j] = sum;
            if (sum > max) max = sum;
        }
        float expSum = 0.0f;
        for (int j = 0; j < outputProbs.length; j++) {
            outputProbs[j] = (float) Math.exp(outputProbs[j] - max);
            expSum += outputProbs[j];
        }
        for (int j = 0; j < outputProbs.length; j++) {
            outputProbs[j] /= expSum;
        }
    }

    private void computeSigmoid(float[] hidden, float[][] weights, float[] outputProbs) {
        for (int j = 0; j < outputProbs.length; j++) {
            float sum = 0.0f;
            for (int i = 0; i < hidden.length; i++) {
                sum += hidden[i] * weights[i][j];
            }
            outputProbs[j] = 1.0f / (1.0f + (float) Math.exp(-sum));
        }
    }

    public int sampleCategorical(float[] probs) {
        float r = random.nextFloat();
        float cumulative = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1;
    }

    // Getters for disk serialization
    public float[][] getWTrunk() { return wTrunk; }
    public float[][] getWMove() { return wMove; }
    public float[][] getWToggle() { return wToggle; }
    public float[][] getWYaw() { return wYaw; }
    public float[][] getWPitch() { return wPitch; }
}
