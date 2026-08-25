package com.example.newgen6.rl.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Allocation-conscious Adam optimizer for Java float parameters.
 *
 * Parameters and gradients are registered explicitly. The optimizer owns
 * the first/second moments and applies bias-corrected Adam updates.
 */
public final class AdamOptimizer {

    private static final class Entry {
        final float[] parameter;
        final float[] gradient;
        final float[] firstMoment;
        final float[] secondMoment;

        Entry(float[] parameter, float[] gradient) {
            if (parameter == null || gradient == null) {
                throw new IllegalArgumentException("Parameter/gradient cannot be null");
            }
            if (parameter.length != gradient.length) {
                throw new IllegalArgumentException(
                        "Parameter/gradient size mismatch: "
                                + parameter.length + " != " + gradient.length
                );
            }

            this.parameter = parameter;
            this.gradient = gradient;
            this.firstMoment = new float[parameter.length];
            this.secondMoment = new float[parameter.length];
        }
    }

    private final List<Entry> entries = new ArrayList<>();

    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private long step;

    private double lastGlobalGradNorm;

    public AdamOptimizer(
            float learningRate,
            float beta1,
            float beta2,
            float epsilon) {

        if (!(learningRate > 0.0f)) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        if (!(beta1 >= 0.0f && beta1 < 1.0f)) {
            throw new IllegalArgumentException("beta1 must be in [0, 1)");
        }
        if (!(beta2 >= 0.0f && beta2 < 1.0f)) {
            throw new IllegalArgumentException("beta2 must be in [0, 1)");
        }
        if (!(epsilon > 0.0f)) {
            throw new IllegalArgumentException("epsilon must be > 0");
        }

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * Registers a vector parameter and its gradient.
     *
     * Call this once during network initialization.
     */
    public void register(float[] parameter, float[] gradient) {
        entries.add(new Entry(parameter, gradient));
    }

    /**
     * Registers every row in a matrix parameter.
     */
    public void register(float[][] parameter, float[][] gradient) {
        if (parameter == null || gradient == null) {
            throw new IllegalArgumentException("Matrix cannot be null");
        }

        if (parameter.length != gradient.length) {
            throw new IllegalArgumentException(
                    "Matrix row count mismatch: "
                            + parameter.length + " != " + gradient.length
            );
        }

        for (int i = 0; i < parameter.length; i++) {
            register(parameter[i], gradient[i]);
        }
    }

    /**
     * Removes all registered parameters and optimizer moments.
     */
    public void clearParameters() {
        entries.clear();
        step = 0L;
        lastGlobalGradNorm = 0.0;
    }

    /**
     * Clips all registered gradients by one global L2 norm.
     */
    public double clipGlobalNorm(double maxNorm) {
        if (!(maxNorm > 0.0)) {
            throw new IllegalArgumentException("maxNorm must be > 0");
        }

        double squaredNorm = 0.0;

        for (Entry entry : entries) {
            for (float g : entry.gradient) {
                if (Float.isFinite(g)) {
                    squaredNorm += (double) g * g;
                }
            }
        }

        double norm = Math.sqrt(squaredNorm);
        lastGlobalGradNorm = norm;

        if (!Double.isFinite(norm) || norm <= maxNorm) {
            if (!Double.isFinite(norm)) {
                zeroAllGradients();
            }
            return lastGlobalGradNorm;
        }

        float scale = (float) (maxNorm / Math.max(norm, 1.0e-30));

        for (Entry entry : entries) {
            for (int i = 0; i < entry.gradient.length; i++) {
                entry.gradient[i] *= scale;
            }
        }

        lastGlobalGradNorm = maxNorm;
        return lastGlobalGradNorm;
    }

    /**
     * Performs one Adam optimization step.
     */
    public void step() {
        step++;

        double biasCorrection1 = 1.0 - Math.pow(beta1, step);
        double biasCorrection2 = 1.0 - Math.pow(beta2, step);

        if (!(biasCorrection1 > 0.0) || !(biasCorrection2 > 0.0)) {
            throw new IllegalStateException("Invalid Adam bias correction");
        }

        double stepSize = learningRate
                * Math.sqrt(biasCorrection2)
                / biasCorrection1;

        for (Entry entry : entries) {
            for (int i = 0; i < entry.parameter.length; i++) {
                float gradient = entry.gradient[i];

                if (!Float.isFinite(gradient)) {
                    gradient = 0.0f;
                }

                float m = entry.firstMoment[i];
                float v = entry.secondMoment[i];

                m = beta1 * m + (1.0f - beta1) * gradient;
                v = beta2 * v + (1.0f - beta2) * gradient * gradient;

                entry.firstMoment[i] = m;
                entry.secondMoment[i] = v;

                double denominator = Math.sqrt(Math.max(0.0, v)) + epsilon;
                double update = stepSize * m / denominator;

                if (Double.isFinite(update)) {
                    float newValue = (float) (entry.parameter[i] - update);

                    if (Float.isFinite(newValue)) {
                        entry.parameter[i] = newValue;
                    }
                }

                entry.gradient[i] = 0.0f;
            }
        }
    }

    public double globalGradNorm() {
        return lastGlobalGradNorm;
    }

    public long stepCount() {
        return step;
    }

    public float learningRate() {
        return learningRate;
    }

    public float beta1() {
        return beta1;
    }

    public float beta2() {
        return beta2;
    }

    public float epsilon() {
        return epsilon;
    }

    private void zeroAllGradients() {
        for (Entry entry : entries) {
            Arrays.fill(entry.gradient, 0.0f);
        }
        lastGlobalGradNorm = 0.0;
    }
}