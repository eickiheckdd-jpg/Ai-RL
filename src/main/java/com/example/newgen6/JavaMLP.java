package com.example.newgen6;

import java.util.Random;

public class JavaMLP {
    private final int[] layerSizes;
    private final float[][] weights;
    private final float[][] biases;
    
    private final float[][] mWeights, vWeights, mBiases, vBiases;
    private final float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f, learningRate;
    private int t = 0;

    private float[][] activations;
    private float[][] zValues;

    public JavaMLP(int inputSize, int hidden1, int hidden2, int outputSize, float learningRate) {
        this.layerSizes = new int[]{inputSize, hidden1, hidden2, outputSize};
        this.learningRate = learningRate;
        
        int numLayers = layerSizes.length - 1;
        weights = new float[numLayers][];
        biases = new float[numLayers][];
        mWeights = new float[numLayers][]; vWeights = new float[numLayers][];
        mBiases = new float[numLayers][]; vBiases = new float[numLayers][];
        
        Random rand = new Random();
        
        for (int i = 0; i < numLayers; i++) {
            int in = layerSizes[i];
            int out = layerSizes[i + 1];
            weights[i] = new float[in * out];
            biases[i] = new float[out];
            
            mWeights[i] = new float[in * out]; vWeights[i] = new float[in * out];
            mBiases[i] = new float[out]; vBiases[i] = new float[out];
            
            float stddev = (float) Math.sqrt(2.0 / in);
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (float) rand.nextGaussian() * stddev;
            }
        }
    }

    public float[] forward(float[] input) {
        activations = new float[layerSizes.length][];
        zValues = new float[layerSizes.length - 1][];
        activations[0] = input.clone();

        for (int i = 0; i < layerSizes.length - 1; i++) {
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
                activations[i + 1][o] = (i == layerSizes.length - 2) ? sum : Math.max(0, sum);
            }
        }
        return activations[layerSizes.length - 1];
    }

    public void trainStep(float[] input, float[] targetVector, float maxGradient) {
        forward(input);
        t++;
        
        int numLayers = layerSizes.length - 1;
        float[][] delta = new float[numLayers][];
        
        delta[numLayers - 1] = new float[layerSizes[numLayers]];
        for (int i = 0; i < layerSizes[numLayers]; i++) {
            float error = activations[numLayers][i] - targetVector[i];
            if (Math.abs(error) <= 1.0f) {
                delta[numLayers - 1][i] = error;
            } else {
                delta[numLayers - 1][i] = Math.signum(error);
            }
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

        for (int l = 0; l < numLayers; l++) {
            int in = layerSizes[l];
            int out = layerSizes[l + 1];
            
            for (int o = 0; o < out; o++) {
                float gradB = Math.max(-maxGradient, Math.min(maxGradient, delta[l][o]));
                mBiases[l][o] = beta1 * mBiases[l][o] + (1 - beta1) * gradB;
                vBiases[l][o] = beta2 * vBiases[l][o] + (1 - beta2) * (gradB * gradB);
                float mHatB = mBiases[l][o] / (1 - (float) Math.pow(beta1, t));
                float vHatB = vBiases[l][o] / (1 - (float) Math.pow(beta2, t));
                biases[l][o] -= learningRate * mHatB / ((float) Math.sqrt(vHatB) + epsilon);

                for (int i = 0; i < in; i++) {
                    float gradW = delta[l][o] * activations[l][i];
                    gradW = Math.max(-maxGradient, Math.min(maxGradient, gradW));
                    
                    int wIdx = i * out + o;
                    mWeights[l][wIdx] = beta1 * mWeights[l][wIdx] + (1 - beta1) * gradW;
                    vWeights[l][wIdx] = beta2 * vWeights[l][wIdx] + (1 - beta2) * (gradW * gradW);
                    
                    float mHatW = mWeights[l][wIdx] / (1 - (float) Math.pow(beta1, t));
                    float vHatW = vWeights[l][wIdx] / (1 - (float) Math.pow(beta2, t));
                    
                    weights[l][wIdx] -= learningRate * mHatW / ((float) Math.sqrt(vHatW) + epsilon);
                }
            }
        }
    }

    public void copyWeightsFrom(JavaMLP other) {
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(other.weights[i], 0, this.weights[i], 0, this.weights[i].length);
            System.arraycopy(other.biases[i], 0, this.biases[i], 0, this.biases[i].length);
        }
    }

    public float[][] getWeights() { return weights; }
    public float[][] getBiases() { return biases; }
}
