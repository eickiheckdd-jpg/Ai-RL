package com.example.newgen6.rl.nn;

import java.util.Random;

/**
 * Numerically stable categorical-distribution utilities.
 *
 * Logits are converted to probabilities with a max-subtracted softmax.
 * No external ML library is required.
 */
public final class Categorical {

    private static final double MIN_PROBABILITY = 1.0e-12;

    private Categorical() {
    }

    public static double[] softmax(double[] logits) {
        validateLogits(logits);

        double max = logits[0];

        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        double sum = 0.0;

        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(clampExponent(logits[i] - max));

            if (!Double.isFinite(e)) {
                e = 0.0;
            }

            sum += e;
        }

        if (!(sum > 0.0) || !Double.isFinite(sum)) {
            return uniform(logits.length);
        }

        double[] probabilities = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = logits[i] == Double.NEGATIVE_INFINITY
                    ? 0.0
                    : Math.max(0.0, eSafe(logits[i] - max) / sum);
        }

        normalize(probabilities);
        return probabilities;
    }

    public static int sample(double[] logits, Random rng) {
        if (rng == null) {
            throw new IllegalArgumentException("rng cannot be null");
        }

        double[] probabilities = softmax(logits);

        double r = rng.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];

            if (r < cumulative || i == probabilities.length - 1) {
                return i;
            }
        }

        return probabilities.length - 1;
    }

    public static double logProb(double[] logits, int action) {
        validateLogits(logits);

        if (action < 0 || action >= logits.length) {
            throw new IllegalArgumentException(
                    "Action " + action
                            + " outside categorical range [0, "
                            + (logits.length - 1) + "]"
            );
        }

        double max = logits[0];

        for (int i = 1; i < logits.length; i++) {
            max = Math.max(max, logits[i]);
        }

        double sumExp = 0.0;

        for (double logit : logits) {
            sumExp += Math.exp(clampExponent(logit - max));
        }

        if (!(sumExp > 0.0) || !Double.isFinite(sumExp)) {
            return -Math.log(logits.length);
        }

        return logits[action] - max - Math.log(sumExp);
    }

    public static double entropy(double[] logits) {
        double[] probabilities = softmax(logits);

        double entropy = 0.0;

        for (double p : probabilities) {
            if (p > MIN_PROBABILITY) {
                entropy -= p * Math.log(p);
            }
        }

        return Double.isFinite(entropy) ? entropy : 0.0;
    }

    public static double[] logSoftmax(double[] logits) {
        validateLogits(logits);

        double max = logits[0];

        for (int i = 1; i < logits.length; i++) {
            max = Math.max(max, logits[i]);
        }

        double sumExp = 0.0;

        for (double logit : logits) {
            sumExp += Math.exp(clampExponent(logit - max));
        }

        if (!(sumExp > 0.0) || !Double.isFinite(sumExp)) {
            double uniformLogProb = -Math.log(logits.length);
            double[] result = new double[logits.length];

            for (int i = 0; i < result.length; i++) {
                result[i] = uniformLogProb;
            }

            return result;
        }

        double logSumExp = max + Math.log(sumExp);
        double[] result = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            result[i] = logits[i] - logSumExp;
        }

        return result;
    }

    private static void validateLogits(double[] logits) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException(
                    "Categorical logits cannot be null or empty"
            );
        }

        for (double logit : logits) {
            if (Double.isNaN(logit) || logit == Double.POSITIVE_INFINITY) {
                throw new IllegalArgumentException(
                        "Categorical logits contain invalid value: " + logit
                );
            }
        }
    }

    private static double eSafe(double x) {
        if (x < -745.0) {
            return 0.0;
        }

        if (x > 709.0) {
            return Double.MAX_VALUE;
        }

        double result = Math.exp(x);
        return Double.isFinite(result) ? result : 0.0;
    }

    private static double clampExponent(double x) {
        return Math.max(-745.0, Math.min(709.0, x));
    }

    private static double[] uniform(int size) {
        double[] result = new double[size];
        double p = 1.0 / size;

        for (int i = 0; i < size; i++) {
            result[i] = p;
        }

        return result;
    }

    private static void normalize(double[] probabilities) {
        double sum = 0.0;

        for (double p : probabilities) {
            sum += p;
        }

        if (!(sum > 0.0) || !Double.isFinite(sum)) {
            double p = 1.0 / probabilities.length;

            for (int i = 0; i < probabilities.length; i++) {
                probabilities[i] = p;
            }

            return;
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
    }
}