package com.example.newgen6.rl.nn;

/**
 * Minimal layer contract for the native float[] autograd-lite engine.
 * Every layer caches whatever it needs from forward() to compute backward().
 */
public interface Layer {
    float[] forward(float[] input);

    /** gradOutput = dLoss/dOutput. Returns dLoss/dInput. */
    float[] backward(float[] gradOutput);
}
