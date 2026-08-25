package com.example.newgen6.rl.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Fully-connected linear layer: y = W*x + b, weights laid out as [out][in].
 *
 * Each Dense instance owns its own gradient accumulators and Adam moment
 * buffers, so the PPOTrainer can treat "all Dense layers in the network" as
 * a flat, self-contained parameter set without any external bookkeeping.
 */
public class Dense implements Layer {

    public final int inputSize;
    public final int outputSize;

    private final float[][] weights;   // [out][in]
    private final float[] bias;        // [out]

    // Gradient accumulators, summed across a minibatch. Call zeroGrad()
    // before starting a new minibatch accumulation.
    private final float[][] gradWeights;
    private final float[] gradBias;

    // Adam moment estimates (persist across steps).
    private final float[][] mWeights, vWeights;
    private final float[] mBias, vBias;

    private float[] lastInput;

    public Dense(int inputSize, int outputSize, Random rng) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new float[outputSize][inputSize];
        this.bias = new float[outputSize];
        this.gradWeights = new float[outputSize][inputSize];
        this.gradBias = new float[outputSize];
        this.mWeights = new float[outputSize][inputSize];
        this.vWeights = new float[outputSize][inputSize];
        this.mBias = new float[outputSize];
        this.vBias = new float[outputSize];

        // Xavier/Glorot-ish uniform init.
        float limit = (float) Math.sqrt(6.0 / (inputSize + outputSize));
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] = (float) ((rng.nextDouble() * 2 - 1) * limit);
            }
        }
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                    "Dense expected input size " + inputSize + " but got " + input.length);
        }
        this.lastInput = input;
        float[] out = new float[outputSize];
        for (int o = 0; o < outputSize; o++) {
            float sum = bias[o];
            float[] row = weights[o];
            for (int i = 0; i < inputSize; i++) {
                sum += row[i] * input[i];
            }
            out[o] = sum;
        }
        return out;
    }

    @Override
    public float[] backward(float[] gradOutput) {
        float[] gradInput = new float[inputSize];
        for (int o = 0; o < outputSize; o++) {
            float go = gradOutput[o];
            if (go == 0f) continue;
            float[] wRow = weights[o];
            float[] gwRow = gradWeights[o];
            for (int i = 0; i < inputSize; i++) {
                gwRow[i] += go * lastInput[i];
                gradInput[i] += go * wRow[i];
            }
            gradBias[o] += go;
        }
        return gradInput;
    }

    public void zeroGrad() {
        for (float[] row : gradWeights) Arrays.fill(row, 0f);
        Arrays.fill(gradBias, 0f);
    }

    /** Scales accumulated gradients in place (used to average a minibatch sum). */
    public void scaleGrad(float factor) {
        for (float[] row : gradWeights) {
            for (int i = 0; i < row.length; i++) row[i] *= factor;
        }
        for (int o = 0; o < gradBias.length; o++) gradBias[o] *= factor;
    }

    public float[][] weights() { return weights; }
    public float[] bias() { return bias; }
    public float[][] gradWeights() { return gradWeights; }
    public float[] gradBias() { return gradBias; }
    public float[][] mWeights() { return mWeights; }
    public float[][] vWeights() { return vWeights; }
    public float[] mBias() { return mBias; }
    public float[] vBias() { return vBias; }

    public void writeTo(DataOutputStream out) throws IOException {
        for (float[] row : weights) for (float w : row) out.writeFloat(w);
        for (float b : bias) out.writeFloat(b);
    }

    public void readFrom(DataInputStream in) throws IOException {
        for (int o = 0; o < outputSize; o++)
            for (int i = 0; i < inputSize; i++)
                weights[o][i] = in.readFloat();
        for (int o = 0; o < outputSize; o++)
            bias[o] = in.readFloat();
    }
}
