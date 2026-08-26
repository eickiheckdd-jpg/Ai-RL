package com.example.newgen6.rl;

import java.util.Random;

public class PolicyNetwork {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    // Trainable Parameters
    private final float[][] w1; 
    private final float[] b1;   
    private final float[][] w2; 
    private final float[] b2;   
    private final float[] logStd;

    // Cache Arrays (Prevents object recreation during inference)
    private final float[] hiddenActivation;
    private final float[] actionMeans;
    private final float[] sampledActions;

    private final Random random = new Random();

    public PolicyNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.w1 = new float[inputSize][hiddenSize];
        this.b1 = new float[hiddenSize];
        this.w2 = new float[hiddenSize][outputSize];
        this.b2 = new float[outputSize];
        this.logStd = new float[outputSize];

        this.hiddenActivation = new float[hiddenSize];
        this.actionMeans = new float[outputSize];
        this.sampledActions = new float[outputSize];

        initializeXavier();
    }

    private void initializeXavier() {
        float limit1 = (float) Math.sqrt(6.0 / (inputSize + hiddenSize));
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                w1[i][j] = (random.nextFloat() * 2.0f - 1.0f) * limit1;
            }
        }

        float limit2 = (float) Math.sqrt(6.0 / (hiddenSize + outputSize));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                w2[i][j] = (random.nextFloat() * 2.0f - 1.0f) * limit2;
            }
        }

        // Start with moderate action variance (~0.5 std deviation)
        for (int i = 0; i < outputSize; i++) {
            logStd[i] = -0.693f; 
        }
    }

    public float[] forward(float[] state) {
        // 1. Input -> Hidden Layer (ReLU)
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += state[i] * w1[i][j];
            }
            hiddenActivation[j] = Math.max(0.0f, sum); 
        }

        // 2. Hidden -> Output Layer (Tanh ensures output is strictly [-1.0, 1.0])
        for (int j = 0; j < outputSize; j++) {
            float sum = b2[j];
            for (int i = 0; i < hiddenSize; i++) {
                sum += hiddenActivation[i] * w2[i][j];
            }
            actionMeans[j] = (float) Math.tanh(sum);
        }

        return actionMeans;
    }

    public float[] sampleAction(float[] state) {
        forward(state);

        for (int i = 0; i < outputSize; i++) {
            float std = (float) Math.exp(logStd[i]);
            float noise = (float) random.nextGaussian();
            float action = actionMeans[i] + (std * noise);

            // Enforce hard clamp so network actions never break Minecraft limits
            sampledActions[i] = Math.max(-1.0f, Math.min(1.0f, action));
        }

        return sampledActions;
    }

    // Accessors for PPO Trainer Backpropagation
    public float[][] getW2() { return w2; }
}
