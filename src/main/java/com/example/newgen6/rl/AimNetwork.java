package com.example.newgen6.rl;

import java.util.Random;

/**
 * Small Java-only MLP with separate Gaussian policy and value heads.
 * All parameters are randomly initialized.
 */
public final class AimNetwork {
    public static final int INPUTS = Observation.SIZE;
    public static final int HIDDEN = 32;
    public static final int OUTPUTS = 5; // yaw mean, pitch mean, yaw log std, pitch log std, value

    private final double[][] weights1 = new double[HIDDEN][INPUTS];
    private final double[] bias1 = new double[HIDDEN];
    private final double[][] weights2 = new double[OUTPUTS][HIDDEN];
    private final double[] bias2 = new double[OUTPUTS];

    private final double[][] gradW1 = new double[HIDDEN][INPUTS];
    private final double[] gradB1 = new double[HIDDEN];
    private final double[][] gradW2 = new double[OUTPUTS][HIDDEN];
    private final double[] gradB2 = new double[OUTPUTS];

    private final double[][] adamMW1 = new double[HIDDEN][INPUTS];
    private final double[][] adamVW1 = new double[HIDDEN][INPUTS];
    private final double[] adamMB1 = new double[HIDDEN];
    private final double[] adamVB1 = new double[HIDDEN];
    private final double[][] adamMW2 = new double[OUTPUTS][HIDDEN];
    private final double[][] adamVW2 = new double[OUTPUTS][HIDDEN];
    private final double[] adamMB2 = new double[OUTPUTS];
    private final double[] adamVB2 = new double[OUTPUTS];

    private final Random random;
    private long optimizerStep;

    public AimNetwork(long seed) {
        random = new Random(seed);
        initialize();
    }

    private void initialize() {
        double limit1 = Math.sqrt(6.0 / (INPUTS + HIDDEN));
        double limit2 = Math.sqrt(6.0 / (HIDDEN + OUTPUTS));

        for (int h = 0; h < HIDDEN; h++) {
            for (int i = 0; i < INPUTS; i++) {
                weights1[h][i] = uniform(-limit1, limit1);
            }
            bias1[h] = 0.0;
        }

        for (int o = 0; o < OUTPUTS; o++) {
            for (int h = 0; h < HIDDEN; h++) {
                weights2[o][h] = uniform(-limit2, limit2);
            }
            bias2[o] = 0.0;
        }

        // Start with a deliberately broad but finite Gaussian.
        bias2[2] = Math.log(0.35);
        bias2[3] = Math.log(0.35);
    }

    private double uniform(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }

    public Forward forward(float[] input) {
        if (input.length != INPUTS) {
            throw new IllegalArgumentException("Expected " + INPUTS + " inputs, got " + input.length);
        }

        double[] hiddenPre = new double[HIDDEN];
        double[] hidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            double sum = bias1[h];
            for (int i = 0; i < INPUTS; i++) {
                sum += weights1[h][i] * input[i];
            }
            hiddenPre[h] = sum;
            hidden[h] = Math.max(0.0, sum);
        }

        double[] output = new double[OUTPUTS];
        for (int o = 0; o < OUTPUTS; o++) {
            double sum = bias2[o];
            for (int h = 0; h < HIDDEN; h++) {
                sum += weights2[o][h] * hidden[h];
            }
            output[o] = sum;
        }

        double yawMean = clamp(output[0], -2.0, 2.0);
        double pitchMean = clamp(output[1], -2.0, 2.0);
        double yawLogStd = clamp(output[2], -3.0, 0.5);
        double pitchLogStd = clamp(output[3], -3.0, 0.5);

        return new Forward(input, hiddenPre, hidden, yawMean, pitchMean, yawLogStd, pitchLogStd, output[4]);
    }

    public double value(float[] input) {
        return forward(input).value;
    }

    public Sample sample(float[] input) {
        Forward forward = forward(input);
        double yawStd = Math.exp(forward.yawLogStd);
        double pitchStd = Math.exp(forward.pitchLogStd);
        double rawYaw = forward.yawMean + yawStd * random.nextGaussian();
        double rawPitch = forward.pitchMean + pitchStd * random.nextGaussian();
        double yawAction = Math.tanh(rawYaw);
        double pitchAction = Math.tanh(rawPitch);
        double logProbability = logProbability(forward, rawYaw, rawPitch, yawAction, pitchAction);
        return new Sample((float) yawAction, (float) pitchAction, rawYaw, rawPitch, logProbability, forward.value);
    }

    public double logProbability(float[] input, double rawYaw, double rawPitch) {
        Forward forward = forward(input);
        return logProbability(
                forward,
                rawYaw,
                rawPitch,
                Math.tanh(rawYaw),
                Math.tanh(rawPitch));
    }

    private static double logProbability(
            Forward forward,
            double rawYaw,
            double rawPitch,
            double yawAction,
            double pitchAction) {
        double logp = gaussianLogProbability(rawYaw, forward.yawMean, forward.yawLogStd)
                + gaussianLogProbability(rawPitch, forward.pitchMean, forward.pitchLogStd);

        double yawCorrection = Math.log(Math.max(1e-6, 1.0 - yawAction * yawAction));
        double pitchCorrection = Math.log(Math.max(1e-6, 1.0 - pitchAction * pitchAction));
        return logp - yawCorrection - pitchCorrection;
    }

    private static double gaussianLogProbability(double x, double mean, double logStd) {
        double variance = Math.exp(2.0 * logStd);
        double diff = x - mean;
        return -0.5 * (diff * diff / variance + 2.0 * logStd + Math.log(2.0 * Math.PI));
    }

    public void clearGradients() {
        for (int h = 0; h < HIDDEN; h++) {
            java.util.Arrays.fill(gradW1[h], 0.0);
            gradB1[h] = 0.0;
        }
        for (int o = 0; o < OUTPUTS; o++) {
            java.util.Arrays.fill(gradW2[o], 0.0);
            gradB2[o] = 0.0;
        }
    }

    public void accumulateEntropyGradient(float[] input, double entropyLossCoefficient) {
        Forward f = forward(input);
        double[] dOutput = new double[OUTPUTS];
        // Entropy H = log(sigma) + constant for each independent Gaussian dimension.
        // Loss includes -entropyCoefficient * H, so dL/d(logStd) = -entropyCoefficient.
        dOutput[2] = entropyLossCoefficient;
        dOutput[3] = entropyLossCoefficient;

        for (int o = 0; o < OUTPUTS; o++) {
            if (dOutput[o] == 0.0) {
                continue;
            }
            gradB2[o] += dOutput[o];
            for (int h = 0; h < HIDDEN; h++) {
                gradW2[o][h] += dOutput[o] * f.hidden[h];
            }
        }

        double[] dHidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            dHidden[h] = weights2[2][h] * dOutput[2] + weights2[3][h] * dOutput[3];
            if (f.hiddenPre[h] <= 0.0) {
                dHidden[h] = 0.0;
            }
        }

        for (int h = 0; h < HIDDEN; h++) {
            gradB1[h] += dHidden[h];
            for (int i = 0; i < INPUTS; i++) {
                gradW1[h][i] += dHidden[h] * input[i];
            }
        }
    }

    /**
     * Accumulates one PPO policy/value sample gradient into the network.
     * The caller supplies the derivative of the loss with respect to log-probability
     * and the derivative with respect to value.
     */
    public void accumulateGradient(
            float[] input,
            double rawYaw,
            double rawPitch,
            double dLossDLogProbability,
            double dLossDValue) {
        Forward f = forward(input);

        double yawVariance = Math.exp(2.0 * f.yawLogStd);
        double pitchVariance = Math.exp(2.0 * f.pitchLogStd);
        double yawDiff = rawYaw - f.yawMean;
        double pitchDiff = rawPitch - f.pitchMean;

        double dLogProbDMeanYaw = yawDiff / yawVariance;
        double dLogProbDLogStdYaw = (yawDiff * yawDiff / yawVariance) - 1.0;
        double dLogProbDMeanPitch = pitchDiff / pitchVariance;
        double dLogProbDLogStdPitch = (pitchDiff * pitchDiff / pitchVariance) - 1.0;

        double[] dOutput = new double[OUTPUTS];
        dOutput[0] = dLossDLogProbability * dLogProbDMeanYaw;
        dOutput[1] = dLossDLogProbability * dLogProbDMeanPitch;
        dOutput[2] = dLossDLogProbability * dLogProbDLogStdYaw;
        dOutput[3] = dLossDLogProbability * dLogProbDLogStdPitch;
        dOutput[4] = dLossDValue;

        for (int o = 0; o < OUTPUTS; o++) {
            gradB2[o] += dOutput[o];
            for (int h = 0; h < HIDDEN; h++) {
                gradW2[o][h] += dOutput[o] * f.hidden[h];
            }
        }

        double[] dHidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            double sum = 0.0;
            for (int o = 0; o < OUTPUTS; o++) {
                sum += weights2[o][h] * dOutput[o];
            }
            dHidden[h] = f.hiddenPre[h] > 0.0 ? sum : 0.0;
        }

        for (int h = 0; h < HIDDEN; h++) {
            gradB1[h] += dHidden[h];
            for (int i = 0; i < INPUTS; i++) {
                gradW1[h][i] += dHidden[h] * input[i];
            }
        }
    }

    public void stepAdam(double learningRate, double gradientScale) {
        optimizerStep++;
        double b1Correction = 1.0 - Math.pow(0.9, optimizerStep);
        double b2Correction = 1.0 - Math.pow(0.999, optimizerStep);

        for (int h = 0; h < HIDDEN; h++) {
            for (int i = 0; i < INPUTS; i++) {
                double g = clipGradient(gradW1[h][i] * gradientScale);
                adamMW1[h][i] = 0.9 * adamMW1[h][i] + 0.1 * g;
                adamVW1[h][i] = 0.999 * adamVW1[h][i] + 0.001 * g * g;
                weights1[h][i] -= learningRate
                        * (adamMW1[h][i] / b1Correction)
                        / (Math.sqrt(adamVW1[h][i] / b2Correction) + 1e-8);
            }

            double gb = clipGradient(gradB1[h] * gradientScale);
            adamMB1[h] = 0.9 * adamMB1[h] + 0.1 * gb;
            adamVB1[h] = 0.999 * adamVB1[h] + 0.001 * gb * gb;
            bias1[h] -= learningRate
                    * (adamMB1[h] / b1Correction)
                    / (Math.sqrt(adamVB1[h] / b2Correction) + 1e-8);
        }

        for (int o = 0; o < OUTPUTS; o++) {
            for (int h = 0; h < HIDDEN; h++) {
                double g = clipGradient(gradW2[o][h] * gradientScale);
                adamMW2[o][h] = 0.9 * adamMW2[o][h] + 0.1 * g;
                adamVW2[o][h] = 0.999 * adamVW2[o][h] + 0.001 * g * g;
                weights2[o][h] -= learningRate
                        * (adamMW2[o][h] / b1Correction)
                        / (Math.sqrt(adamVW2[o][h] / b2Correction) + 1e-8);
            }

            double gb = clipGradient(gradB2[o] * gradientScale);
            adamMB2[o] = 0.9 * adamMB2[o] + 0.1 * gb;
            adamVB2[o] = 0.999 * adamVB2[o] + 0.001 * gb * gb;
            bias2[o] -= learningRate
                    * (adamMB2[o] / b1Correction)
                    / (Math.sqrt(adamVB2[o] / b2Correction) + 1e-8);
        }
    }

    public double parameterChecksum() {
        double sum = 0.0;
        for (double[] row : weights1) {
            for (double value : row) {
                sum += value;
            }
        }
        for (double value : bias1) {
            sum += value;
        }
        for (double[] row : weights2) {
            for (double value : row) {
                sum += value;
            }
        }
        for (double value : bias2) {
            sum += value;
        }
        return sum;
    }

    private static double clipGradient(double value) {
        return clamp(value, -10.0, 10.0);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public record Forward(
            float[] input,
            double[] hiddenPre,
            double[] hidden,
            double yawMean,
            double pitchMean,
            double yawLogStd,
            double pitchLogStd,
            double value) {
    }

    public record Sample(
            float yawAction,
            float pitchAction,
            double rawYaw,
            double rawPitch,
            double logProbability,
            double value) {
    }
}
