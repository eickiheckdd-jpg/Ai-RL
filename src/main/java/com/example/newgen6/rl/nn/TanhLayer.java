package com.example.newgen6.rl.nn;

public class TanhLayer implements Layer {

    private float[] lastOutput;

    @Override
    public float[] forward(float[] input) {
        float[] out = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            out[i] = (float) Math.tanh(input[i]);
        }
        this.lastOutput = out;
        return out;
    }

    @Override
    public float[] backward(float[] gradOutput) {
        float[] gradInput = new float[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            float y = lastOutput[i];
            gradInput[i] = gradOutput[i] * (1f - y * y);
        }
        return gradInput;
    }
}
