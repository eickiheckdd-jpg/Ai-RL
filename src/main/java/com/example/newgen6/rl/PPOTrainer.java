package com.example.newgen6.rl;

import java.util.Arrays;

/**
 * Minimal on-device PPO implementation for the initial continuous aim task.
 */
public final class PPOTrainer {
    private static final int ROLLOUT_SIZE = 256;
    private static final int PPO_EPOCHS = 4;
    private static final double GAMMA = 0.99;
    private static final double GAE_LAMBDA = 0.95;
    private static final double CLIP_RANGE = 0.2;
    private static final double POLICY_LEARNING_RATE = 0.00015;
    private static final double VALUE_COEFFICIENT = 0.5;
    private static final double ENTROPY_COEFFICIENT = 0.002;

    private final AimNetwork network;
    private final Transition[] buffer = new Transition[ROLLOUT_SIZE];

    private int count;
    private long totalSteps;
    private long ppoUpdates;
    private double lastMeanReward;
    private double lastMeanYawError;
    private double lastMeanPitchError;
    private double lastPolicyLoss;
    private double lastValueLoss;
    private double lastEntropy;
    private double lastParameterDelta;

    public PPOTrainer(AimNetwork network) {
        this.network = network;
    }

    public boolean addTransition(
            float[] observation,
            double rawYaw,
            double rawPitch,
            double yawAction,
            double pitchAction,
            double oldLogProbability,
            double oldValue,
            double reward,
            boolean done,
            double yawError,
            double pitchError) {
        if (count >= buffer.length) {
            throw new IllegalStateException("PPO rollout buffer is full");
        }

        buffer[count++] = new Transition(
                Arrays.copyOf(observation, observation.length),
                rawYaw,
                rawPitch,
                yawAction,
                pitchAction,
                oldLogProbability,
                oldValue,
                reward,
                done,
                yawError,
                pitchError);
        totalSteps++;

        return count >= buffer.length || done;
    }

    public void update(double bootstrapValue) {
        if (count == 0) {
            return;
        }

        double parameterChecksumBefore = network.parameterChecksum();
        double[] advantages = new double[count];
        double[] returns = new double[count];

        double gae = 0.0;
        for (int i = count - 1; i >= 0; i--) {
            Transition current = buffer[i];
            double nextValue;
            double nonTerminal;
            if (i == count - 1) {
                nextValue = current.done ? 0.0 : bootstrapValue;
                nonTerminal = current.done ? 0.0 : 1.0;
            } else {
                nextValue = buffer[i + 1].oldValue;
                nonTerminal = current.done ? 0.0 : 1.0;
            }

            double delta = current.reward + GAMMA * nextValue * nonTerminal - current.oldValue;
            gae = delta + GAMMA * GAE_LAMBDA * nonTerminal * gae;
            advantages[i] = gae;
            returns[i] = advantages[i] + current.oldValue;
        }

        normalize(advantages);

        double meanReward = 0.0;
        double meanYawError = 0.0;
        double meanPitchError = 0.0;
        for (int i = 0; i < count; i++) {
            meanReward += buffer[i].reward;
            meanYawError += Math.abs(buffer[i].yawError);
            meanPitchError += Math.abs(buffer[i].pitchError);
        }
        meanReward /= count;
        meanYawError /= count;
        meanPitchError /= count;

        double policyLoss = 0.0;
        double valueLoss = 0.0;
        double entropy = 0.0;

        for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            network.clearGradients();
            double epochPolicyLoss = 0.0;
            double epochValueLoss = 0.0;
            double epochEntropy = 0.0;

            for (int i = 0; i < count; i++) {
                Transition t = buffer[i];
                AimNetwork.Forward forward = network.forward(t.observation);
                double newLogProbability = network.logProbability(t.observation, t.rawYaw, t.rawPitch);
                double ratio = Math.exp(clamp(newLogProbability - t.oldLogProbability, -20.0, 20.0));
                double advantage = advantages[i];

                boolean clipped = (advantage >= 0.0 && ratio > 1.0 + CLIP_RANGE)
                        || (advantage < 0.0 && ratio < 1.0 - CLIP_RANGE);

                double objective = clipped
                        ? ((advantage >= 0.0 ? 1.0 + CLIP_RANGE : 1.0 - CLIP_RANGE) * advantage)
                        : ratio * advantage;
                double dLossDLogProbability = clipped ? 0.0 : -advantage * ratio;

                double valueError = forward.value() - returns[i];
                double valueSampleLoss = 0.5 * valueError * valueError;
                double valueDerivative = VALUE_COEFFICIENT * valueError;

                double sampleEntropy = forward.yawLogStd() + forward.pitchLogStd() + Math.log(2.0 * Math.PI * Math.E);
                network.accumulateGradient(
                        t.observation,
                        t.rawYaw,
                        t.rawPitch,
                        dLossDLogProbability,
                        valueDerivative);
                network.accumulateEntropyGradient(t.observation, -ENTROPY_COEFFICIENT);

                epochPolicyLoss += -objective;
                epochValueLoss += valueSampleLoss;
                epochEntropy += sampleEntropy;
            }

            network.stepAdam(POLICY_LEARNING_RATE, 1.0 / count);
            policyLoss = epochPolicyLoss / count;
            valueLoss = epochValueLoss / count;
            entropy = epochEntropy / count;
        }

        lastMeanReward = meanReward;
        lastMeanYawError = meanYawError;
        lastMeanPitchError = meanPitchError;
        lastPolicyLoss = policyLoss;
        lastValueLoss = valueLoss;
        lastEntropy = entropy;
        lastParameterDelta = network.parameterChecksum() - parameterChecksumBefore;
        ppoUpdates++;

        Arrays.fill(buffer, 0, count, null);
        count = 0;
    }

    private static void normalize(double[] values) {
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean /= values.length;

        double variance = 0.0;
        for (double value : values) {
            double delta = value - mean;
            variance += delta * delta;
        }
        variance /= values.length;
        double scale = 1.0 / Math.sqrt(variance + 1e-8);

        for (int i = 0; i < values.length; i++) {
            values[i] = (values[i] - mean) * scale;
        }
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public int bufferedTransitions() {
        return count;
    }

    public long totalSteps() {
        return totalSteps;
    }

    public long ppoUpdates() {
        return ppoUpdates;
    }

    public double lastMeanReward() {
        return lastMeanReward;
    }

    public double lastMeanYawError() {
        return lastMeanYawError;
    }

    public double lastMeanPitchError() {
        return lastMeanPitchError;
    }

    public double lastPolicyLoss() {
        return lastPolicyLoss;
    }

    public double lastValueLoss() {
        return lastValueLoss;
    }

    public double lastEntropy() {
        return lastEntropy;
    }

    public double lastParameterDelta() {
        return lastParameterDelta;
    }

    private record Transition(
            float[] observation,
            double rawYaw,
            double rawPitch,
            double yawAction,
            double pitchAction,
            double oldLogProbability,
            double oldValue,
            double reward,
            boolean done,
            double yawError,
            double pitchError) {
    }
}
