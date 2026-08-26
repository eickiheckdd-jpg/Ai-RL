package com.example.newgen6.rl;

import com.example.newgen6.rl.nn.DenseLayer;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

/**
 * Defensive validation helpers for the Java-only RL pipeline.
 *
 * This class does not alter gameplay or model parameters.
 */
public final class TrainingValidator {

    private TrainingValidator() {
    }

    public static void validateObservation(float[] observation) {
        requireFinite(observation, "observation");

        if (observation.length != RLConstants.OBSERVATION_SIZE) {
            throw new IllegalArgumentException(
                    "Observation size mismatch: got "
                            + observation.length
                            + ", expected "
                            + RLConstants.OBSERVATION_SIZE
            );
        }
    }

    public static void validateHiddenState(float[] hidden) {
        requireFinite(hidden, "hidden state");

        if (hidden.length != RLConstants.GRU_HIDDEN_SIZE) {
            throw new IllegalArgumentException(
                    "Hidden size mismatch: got "
                            + hidden.length
                            + ", expected "
                            + RLConstants.GRU_HIDDEN_SIZE
            );
        }
    }

    public static void validateReward(float reward) {
        if (!Float.isFinite(reward)) {
            throw new IllegalArgumentException(
                    "Reward contains NaN or Infinity"
            );
        }
    }

    public static void validateAction(Action action) {
        if (action == null) {
            throw new IllegalArgumentException("Action cannot be null");
        }
        action.validate();
    }

    public static void validateOutput(
            PolicyValueNetwork.Output output) {

        if (output == null) {
            throw new IllegalArgumentException(
                    "Policy output cannot be null"
            );
        }

        requireFinite(output.moveLogits, "move logits");
        requireFinite(output.yawLogits, "yaw logits");
        requireFinite(output.pitchLogits, "pitch logits");
        requireFinite(output.jumpLogits, "jump logits");
        requireFinite(output.sprintLogits, "sprint logits");
        requireFinite(output.sneakLogits, "sneak logits");
        requireFinite(output.attackLogits, "attack logits");

        if (!Double.isFinite(output.value)) {
            throw new IllegalArgumentException(
                    "Value output contains NaN or Infinity"
            );
        }

        validateHiddenState(output.hiddenOut);
    }

    public static void validateNetwork(
            PolicyValueNetwork network) {

        if (network == null) {
            throw new IllegalArgumentException(
                    "Network cannot be null"
            );
        }

        validateDenseLayer(network.trunk(), "trunk");
        validateDenseLayer(network.moveHead(), "move head");
        validateDenseLayer(network.yawHead(), "yaw head");
        validateDenseLayer(network.pitchHead(), "pitch head");
        validateDenseLayer(network.jumpHead(), "jump head");
        validateDenseLayer(network.sprintHead(), "sprint head");
        validateDenseLayer(network.sneakHead(), "sneak head");
        validateDenseLayer(network.attackHead(), "attack head");
        validateDenseLayer(network.valueHead(), "value head");

        requireFiniteMatrix(
                network.gru().weightsZ(),
                "GRU Wz"
        );
        requireFiniteMatrix(
                network.gru().weightsR(),
                "GRU Wr"
        );
        requireFiniteMatrix(
                network.gru().weightsN(),
                "GRU Wn"
        );
        requireFiniteMatrix(
                network.gru().recurrentWeightsZ(),
                "GRU Uz"
        );
        requireFiniteMatrix(
                network.gru().recurrentWeightsR(),
                "GRU Ur"
        );
        requireFiniteMatrix(
                network.gru().recurrentWeightsN(),
                "GRU Un"
        );

        requireFinite(
                network.gru().biasZ(),
                "GRU bz"
        );
        requireFinite(
                network.gru().biasR(),
                "GRU br"
        );
        requireFinite(
                network.gru().biasN(),
                "GRU bn"
        );
    }

    private static void validateDenseLayer(
            DenseLayer layer,
            String name) {

        if (layer == null) {
            throw new IllegalArgumentException(
                    name + " layer cannot be null"
            );
        }

        requireFiniteMatrix(
                layer.weights(),
                name + " weights"
        );
        requireFinite(
                layer.bias(),
                name + " bias"
        );
    }

    private static void requireFinite(
            float[] values,
            String name) {

        if (values == null) {
            throw new IllegalArgumentException(
                    name + " cannot be null"
            );
        }

        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                throw new IllegalArgumentException(
                        name
                                + " contains non-finite value at index "
                                + i
                );
            }
        }
    }

    private static void requireFinite(
            double[] values,
            String name) {

        if (values == null) {
            throw new IllegalArgumentException(
                    name + " cannot be null"
            );
        }

        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException(
                        name
                                + " contains non-finite value at index "
                                + i
                );
            }
        }
    }

    private static void requireFiniteMatrix(
            float[][] matrix,
            String name) {

        if (matrix == null) {
            throw new IllegalArgumentException(
                    name + " cannot be null"
            );
        }

        for (int row = 0; row < matrix.length; row++) {
            requireFinite(
                    matrix[row],
                    name + "[" + row + "]"
            );
        }
    }
}