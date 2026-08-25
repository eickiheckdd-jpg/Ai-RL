package com.example.newgen6.rl.nn;

/**
 * Standalone softmax layer with a general-purpose backward pass.
 *
 * The PPO training path (see DiscreteHead) does NOT route through this
 * class's backward() -- it uses a closed-form shortcut for the
 * log-prob + entropy objective (mathematically equivalent, cheaper, and
 * avoids catastrophic cancellation). This class is provided standalone to
 * satisfy the "Softmax layer with forward()/backward()" requirement and for
 * any other use (e.g. plain classification, debugging action probabilities).
 */
public class SoftmaxLayer implements Layer {

    private float[] lastOutput;

    @Override
    public float[] forward(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : input) if (v > max) max = v;

        float[] out = new float[input.length];
        float sum = 0f;
        for (int i = 0; i < input.length; i++) {
            out[i] = (float) Math.exp(input[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;

        this.lastOutput = out;
        return out;
    }

    /**
     * Full Jacobian-vector product: dInput_j = p_j * (gradOutput_j - sum_k gradOutput_k * p_k)
     */
    @Override
    public float[] backward(float[] gradOutput) {
        float dot = 0f;
        for (int k = 0; k < lastOutput.length; k++) dot += gradOutput[k] * lastOutput[k];

        float[] gradInput = new float[gradOutput.length];
        for (int j = 0; j < gradOutput.length; j++) {
            gradInput[j] = lastOutput[j] * (gradOutput[j] - dot);
        }
        return gradInput;
    }
}
