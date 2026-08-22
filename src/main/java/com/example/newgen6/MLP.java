package com.example.newgen6.rl;

import java.util.Random;

/**
 * Minimal dense feedforward network: arbitrary layer sizes, ReLU on hidden
 * layers, linear output (softmax/argmax applied outside for the actor).
 * Pure Java arrays only — no ND4J/DL4J/etc.
 *
 * weights[l] has shape [layerSizes[l]][layerSizes[l+1]]
 * biases[l]  has shape [layerSizes[l+1]]
 */
public class MLP {

    private final int[] layerSizes;
    private final double[][][] weights;
    private final double[][] biases;

    public MLP(int[] layerSizes, long seed) {
        this.layerSizes = layerSizes;
        int numLayers = layerSizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];

        Random rnd = new Random(seed);
        for (int l = 0; l < numLayers; l++) {
            int in = layerSizes[l];
            int out = layerSizes[l + 1];
            weights[l] = new double[in][out];
            biases[l] = new double[out];
            // He initialization, reasonable default for ReLU nets
            double scale = Math.sqrt(2.0 / in);
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    weights[l][i][j] = rnd.nextGaussian() * scale;
                }
            }
        }
    }

    /** Container for a forward pass, needed to run backprop afterward. */
    public static class ForwardCache {
        double[][] preActivations;  // z at each layer (post weights+bias, pre-activation)
        double[][] activations;     // a at each layer, activations[0] = input
    }

    public ForwardCache forwardWithCache(double[] input) {
        int numLayers = weights.length;
        ForwardCache cache = new ForwardCache();
        cache.preActivations = new double[numLayers][];
        cache.activations = new double[numLayers + 1][];
        cache.activations[0] = input;

        double[] a = input;
        for (int l = 0; l < numLayers; l++) {
            int out = layerSizes[l + 1];
            double[] z = new double[out];
            for (int j = 0; j < out; j++) {
                double sum = biases[l][j];
                for (int i = 0; i < a.length; i++) {
                    sum += a[i] * weights[l][i][j];
                }
                z[j] = sum;
            }
            boolean isOutputLayer = (l == numLayers - 1);
            double[] activated = isOutputLayer ? z.clone() : relu(z);

            cache.preActivations[l] = z;
            cache.activations[l + 1] = activated;
            a = activated;
        }
        return cache;
    }

    /** Convenience: forward pass returning only the final output. */
    public double[] forward(double[] input) {
        return forwardWithCache(input).activations[weights.length];
    }

    private static double[] relu(double[] z) {
        double[] out = new double[z.length];
        for (int i = 0; i < z.length; i++) out[i] = Math.max(0.0, z[i]);
        return out;
    }

    /**
     * Accumulates weight/bias gradients for one sample into the provided
     * accumulator arrays (same shape as weights/biases), given the gradient
     * of the loss with respect to the network's final output.
     */
    public void accumulateGradients(ForwardCache cache, double[] outputGradient,
                                     double[][][] gradWAccum, double[][] gradBAccum) {
        int numLayers = weights.length;
        double[] delta = outputGradient; // last layer is linear, so dz = da directly

        for (int l = numLayers - 1; l >= 0; l--) {
            double[] z = cache.preActivations[l];
            boolean isOutputLayer = (l == numLayers - 1);

            double[] dz;
            if (isOutputLayer) {
                dz = delta;
            } else {
                dz = new double[z.length];
                for (int j = 0; j < z.length; j++) {
                    dz[j] = (z[j] > 0.0) ? delta[j] : 0.0; // ReLU derivative
                }
            }

            double[] aPrev = cache.activations[l];
            int in = aPrev.length;
            int out = dz.length;

            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    gradWAccum[l][i][j] += aPrev[i] * dz[j];
                }
            }
            for (int j = 0; j < out; j++) {
                gradBAccum[l][j] += dz[j];
            }

            if (l > 0) {
                double[] deltaPrev = new double[in];
                for (int i = 0; i < in; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < out; j++) {
                        sum += weights[l][i][j] * dz[j];
                    }
                    deltaPrev[i] = sum;
                }
                delta = deltaPrev;
            }
        }
    }

    public double[][][] newGradientAccumulatorW() {
        int numLayers = weights.length;
        double[][][] g = new double[numLayers][][];
        for (int l = 0; l < numLayers; l++) {
            g[l] = new double[weights[l].length][weights[l][0].length];
        }
        return g;
    }

    public double[][] newGradientAccumulatorB() {
        int numLayers = biases.length;
        double[][] g = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) {
            g[l] = new double[biases[l].length];
        }
        return g;
    }

    /** Applies averaged gradients with plain SGD. */
    public void applyGradients(double[][][] gradW, double[][] gradB, int batchSize, double learningRate) {
        double invBatch = 1.0 / Math.max(1, batchSize);
        for (int l = 0; l < weights.length; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                for (int j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] -= learningRate * gradW[l][i][j] * invBatch;
                }
            }
            for (int j = 0; j < biases[l].length; j++) {
                biases[l][j] -= learningRate * gradB[l][j] * invBatch;
            }
        }
    }

    public double[][][] getWeights() { return weights; }
    public double[][] getBiases() { return biases; }
    public int[] getLayerSizes() { return layerSizes; }

    /** Directly overwrite parameters, used when loading saved weights. */
    public void loadParameters(double[][][] newWeights, double[][] newBiases) {
        for (int l = 0; l < weights.length; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                System.arraycopy(newWeights[l][i], 0, weights[l][i], 0, weights[l][i].length);
            }
            System.arraycopy(newBiases[l], 0, biases[l], 0, biases[l].length);
        }
    }
}
