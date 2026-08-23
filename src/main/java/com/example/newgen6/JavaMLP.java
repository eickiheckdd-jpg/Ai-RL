package com.example.newgen6;

import java.util.Random;

public class JavaMLP {
    private final int[] layerSizes;
    private final float[][] weights;
    private final float[][] biases;

    private final float[][] mWeights, vWeights, mBiases, vBiases;
    private final float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f, learningRate;
    private int t = 0;

    public JavaMLP(int inputSize, int hidden1, int hidden2, int outputSize, float learningRate) {
        this.layerSizes = new int[]{inputSize, hidden1, hidden2, outputSize};
        this.learningRate = learningRate;

        int numLayers = layerSizes.length - 1;
        weights = new float[numLayers][];
        biases = new float[numLayers][];
        mWeights = new float[numLayers][]; 
        vWeights = new float[numLayers][];
        mBiases = new float[numLayers][]; 
        vBiases = new float[numLayers][];

        Random rand = new Random();

        for (int i = 0; i < numLayers; i++) {
            int in = layerSizes[i];
            int out = layerSizes[i + 1];
            weights[i] = new float[in * out];
            biases[i] = new float[out];

            mWeights[i] = new float[in * out]; 
            vWeights[i] = new float[in * out];
            mBiases[i] = new float[out]; 
            vBiases[i] = new float[out];

            float stddev = (float) Math.sqrt(2.0 / in);
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (float) rand.nextGaussian() * stddev;
            }
        }
    }

    public synchronized float[] forward(float[] input) {
        float[] current = input.clone();
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int in = layerSizes[i];
            int out = layerSizes[i + 1];
            float[] next = new float[out];

            for (int o = 0; o < out; o++) {
                float sum = biases[i][o];
                for (int j = 0; j < in; j++) {
                    sum += current[j] * weights[i][j * out + o];
                }
                next[o] = (i == layerSizes.length - 2) ? sum : Math.max(0, sum);
            }
            current = next;
        }
        return current;
    }

    /**
     * TRUE MINI-BATCH TRAINING:
     * Accumulates gradients across the entire batch before applying a single Adam update.
     * Dramatically stabilizes learning compared to looping trainStep().
     */
    public synchronized void trainBatchVectorized(float[][] batchInputs, float[][] batchTargets, float maxGradient) {
        int numLayers = layerSizes.length - 1;
        int batchSize = batchInputs.length;
        
        float[][] gradWeights = new float[numLayers][];
        float[][] gradBiases = new float[numLayers][];
        
        for (int i = 0; i < numLayers; i++) {
            gradWeights[i] = new float[weights[i].length];
            gradBiases[i] = new float[biases[i].length];
        }

        // 1. Accumulate Gradients for the whole batch
        for (int b = 0; b < batchSize; b++) {
            float[] input = batchInputs[b];
            float[] targetVector = batchTargets[b];
            
            float[][] activations = new float[layerSizes.length][];
            float[][] zValues = new float[numLayers][];
            activations[0] = input.clone();

            // Forward Pass
            for (int i = 0; i < numLayers; i++) {
                int in = layerSizes[i];
                int out = layerSizes[i + 1];
                zValues[i] = new float[out];
                activations[i + 1] = new float[out];

                for (int o = 0; o < out; o++) {
                    float sum = biases[i][o];
                    for (int j = 0; j < in; j++) {
                        sum += activations[i][j] * weights[i][j * out + o];
                    }
                    zValues[i][o] = sum;
                    activations[i + 1][o] = (i == numLayers - 1) ? sum : Math.max(0, sum);
                }
            }

            // Backward Pass
            float[][] delta = new float[numLayers][];
            delta[numLayers - 1] = new float[layerSizes[numLayers]];

            for (int i = 0; i < layerSizes[numLayers]; i++) {
                float error = activations[numLayers][i] - targetVector[i];
                // Huber loss derivative clipping
                delta[numLayers - 1][i] = (Math.abs(error) <= 1.0f) ? error : Math.signum(error);
            }

            for (int l = numLayers - 2; l >= 0; l--) {
                int out = layerSizes[l + 1];
                int nextOut = layerSizes[l + 2];
                delta[l] = new float[out];

                for (int i = 0; i < out; i++) {
                    float sum = 0;
                    for (int j = 0; j < nextOut; j++) {
                        sum += delta[l + 1][j] * weights[l + 1][i * nextOut + j];
                    }
                    delta[l][i] = zValues[l][i] > 0 ? sum : 0;
                }
            }

            // Accumulate
            for (int l = 0; l < numLayers; l++) {
                int in = layerSizes[l];
                int out = layerSizes[l + 1];

                for (int o = 0; o < out; o++) {
                    gradBiases[l][o] += delta[l][o];
                    for (int i = 0; i < in; i++) {
                        gradWeights[l][i * out + o] += delta[l][o] * activations[l][i];
                    }
                }
            }
        }

        // 2. Apply Adam Optimizer once per batch
        t++;
        float beta1T = 1.0f - (float) Math.pow(beta1, t);
        float beta2T = 1.0f - (float) Math.pow(beta2, t);

        for (int l = 0; l < numLayers; l++) {
            for (int o = 0; o < biases[l].length; o++) {
                float gradB = gradBiases[l][o] / batchSize;
                gradB = Math.max(-maxGradient, Math.min(maxGradient, gradB));
                
                mBiases[l][o] = beta1 * mBiases[l][o] + (1 - beta1) * gradB;
                vBiases[l][o] = beta2 * vBiases[l][o] + (1 - beta2) * (gradB * gradB);

                float mHatB = mBiases[l][o] / beta1T;
                float vHatB = vBiases[l][o] / beta2T;
                biases[l][o] -= learningRate * mHatB / ((float) Math.sqrt(vHatB) + epsilon);
            }
            for (int w = 0; w < weights[l].length; w++) {
                float gradW = gradWeights[l][w] / batchSize;
                gradW = Math.max(-maxGradient, Math.min(maxGradient, gradW));
                
                mWeights[l][w] = beta1 * mWeights[l][w] + (1 - beta1) * gradW;
                vWeights[l][w] = beta2 * vWeights[l][w] + (1 - beta2) * (gradW * gradW);

                float mHatW = mWeights[l][w] / beta1T;
                float vHatW = vWeights[l][w] / beta2T;
                weights[l][w] -= learningRate * mHatW / ((float) Math.sqrt(vHatW) + epsilon);
            }
        }
    }

    // Legacy fallback to guarantee compatibility with your current DoubleDQNAgent code
    public synchronized void trainStep(float[] input, float[] targetVector, float maxGradient) {
        float[][] singleBatchInput = new float[][]{input};
        float[][] singleBatchTarget = new float[][]{targetVector};
        trainBatchVectorized(singleBatchInput, singleBatchTarget, maxGradient);
    }

    public synchronized void copyWeightsFrom(JavaMLP other) {
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(other.weights[i], 0, this.weights[i], 0, this.weights[i].length);
            System.arraycopy(other.biases[i], 0, this.biases[i], 0, this.biases[i].length);
        }
    }

    public float[][] getWeights() { return weights; }
    public float[][] getBiases() { return biases; }
}
