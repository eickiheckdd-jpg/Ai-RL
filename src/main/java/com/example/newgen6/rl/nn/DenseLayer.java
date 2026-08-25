package com.example.newgen6.rl.nn;

import java.util.Random;

/**
 * Small allocation-conscious fully connected layer used by the
 * Java-only RL network.
 *
 * Supports a linear or ReLU activation and stores gradients internally
 * so the optimizer can update the parameters after backpropagation.
 */
public final class DenseLayer {

    public enum Activation {
        NONE,
        RELU
    }

    public static final class Cache {
        float[] input;
        float[] preActivation;
        float[] output;
        boolean valid;
    }

    private final int inputSize;
    private final int outputSize;
    private final Activation activation;

    private final float[][] weights;
    private final float[] bias;

    private final float[][] gradWeights;
    private final float[] gradBias;

    public DenseLayer(
            int inputSize,
            int outputSize,
            Activation activation,
            Random rng) {

        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be > 0");
        }
        if (outputSize <= 0) {
            throw new IllegalArgumentException("outputSize must be > 0");
        }
        if (activation == null) {
            throw new IllegalArgumentException("activation cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("rng cannot be null");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;

        this.weights = new float[outputSize][inputSize];
        this.bias = new float[outputSize];

        this.gradWeights = new float[outputSize][inputSize];
        this.gradBias = new float[outputSize];

        initialize(rng);
    }

    private void initialize(Random rng) {
        /*
         * He initialization is appropriate for ReLU layers.
         * Linear heads use the same bounded scale, which keeps the
         * randomly initialized policy numerically stable.
         */
        float scale = (float) Math.sqrt(2.0 / inputSize);

        for (int o = 0; o < outputSize; o++) {
            for (int j = 0; j < inputSize; j++) {
                weights[o][j] = (float) (rng.nextGaussian() * scale);
            }
            bias[o] = 0.0f;
        }
    }

    public float[] forward(float[] input, Cache cache) {
        checkInput(input);

        float[] output = new float[outputSize];

        for (int o = 0; o < outputSize; o++) {
            float sum = bias[o];

            for (int j = 0; j < inputSize; j++) {
                sum += weights[o][j] * input[j];
            }

            if (activation == Activation.RELU) {
                sum = Math.max(0.0f, sum);
            }

            output[o] = finite(sum);
        }

        if (cache != null) {
            cache.input = input.clone();
            cache.preActivation = new float[outputSize];

            for (int o = 0; o < outputSize; o++) {
                float sum = bias[o];

                for (int j = 0; j < inputSize; j++) {
                    sum += weights[o][j] * input[j];
                }

                cache.preActivation[o] = finite(sum);
            }

            cache.output = output.clone();
            cache.valid = true;
        }

        return output;
    }

    /**
     * Backpropagates dLoss/dOutput and accumulates parameter gradients.
     *
     * @return dLoss/dInput
     */
    public float[] backward(float[] gradOutput, Cache cache) {
        if (cache == null || !cache.valid) {
            throw new IllegalArgumentException(
                    "A valid forward cache is required before backward()"
            );
        }

        if (gradOutput == null || gradOutput.length != outputSize) {
            throw new IllegalArgumentException(
                    "Gradient size mismatch: got "
                            + (gradOutput == null ? "null" : gradOutput.length)
                            + ", expected "
                            + outputSize
            );
        }

        float[] gradInput = new float[inputSize];

        for (int o = 0; o < outputSize; o++) {
            float g = finite(gradOutput[o]);

            if (activation == Activation.RELU &&
                    cache.preActivation[o] <= 0.0f) {
                g = 0.0f;
            }

            gradBias[o] += g;

            for (int j = 0; j < inputSize; j++) {
                gradWeights[o][j] += g * cache.input[j];
                gradInput[j] += g * weights[o][j];
            }
        }

        sanitizeGradients();
        return gradInput;
    }

    public void zeroGrad() {
        for (int o = 0; o < outputSize; o++) {
            for (int j = 0; j < inputSize; j++) {
                gradWeights[o][j] = 0.0f;
            }
            gradBias[o] = 0.0f;
        }
    }

    public int inputSize() {
        return inputSize;
    }

    public int outputSize() {
        return outputSize;
    }

    public Activation activation() {
        return activation;
    }

    public float[][] weights() {
        return weights;
    }

    public float[] bias() {
        return bias;
    }

    public float[][] gradWeights() {
        return gradWeights;
    }

    public float[] gradBias() {
        return gradBias;
    }

    private void checkInput(float[] input) {
        if (input == null) {
            throw new IllegalArgumentException("DenseLayer input cannot be null");
        }

        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                    "DenseLayer input mismatch: got "
                            + input.length
                            + ", expected "
                            + inputSize
            );
        }

        for (float value : input) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "DenseLayer received NaN/Infinity"
                );
            }
        }
    }

    private void sanitizeGradients() {
        for (int o = 0; o < outputSize; o++) {
            if (!Float.isFinite(gradBias[o])) {
                gradBias[o] = 0.0f;
            }

            for (int j = 0; j < inputSize; j++) {
                if (!Float.isFinite(gradWeights[o][j])) {
                    gradWeights[o][j] = 0.0f;
                }
            }
        }
    }

    private static float finite(float value) {
        return Float.isFinite(value) ? value : 0.0f;
    }
}