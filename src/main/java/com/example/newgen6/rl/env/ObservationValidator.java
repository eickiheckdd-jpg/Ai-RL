package com.example.newgen6.rl.env;

/**
 * Validates observations at the environment/NN boundary.
 *
 * This class does not modify observations and does not make
 * gameplay decisions. Its purpose is to catch corrupted or
 * incompatible RL state early.
 */
public final class ObservationValidator {

    private ObservationValidator() {
    }

    /**
     * Performs a strict validation.
     *
     * @throws IllegalArgumentException if the observation is invalid
     */
    public static void validate(Observation observation) {
        if (observation == null) {
            throw new IllegalArgumentException(
                    "Observation cannot be null"
            );
        }

        if (observation.size() != Observation.SIZE) {
            throw new IllegalArgumentException(
                    "Observation dimension mismatch: got "
                            + observation.size()
                            + ", expected "
                            + Observation.SIZE
            );
        }

        if (!observation.isFinite()) {
            throw new IllegalArgumentException(
                    "Observation contains NaN or Infinity"
            );
        }

        if (observation.tick() < 0L) {
            throw new IllegalArgumentException(
                    "Observation tick cannot be negative: "
                            + observation.tick()
            );
        }

        validateValues(observation);
    }

    /**
     * Fast validation intended for the per-tick environment loop.
     *
     * Unlike validate(), this method does not construct detailed
     * error messages for individual feature failures.
     */
    public static boolean isValid(Observation observation) {
        if (observation == null) {
            return false;
        }

        if (observation.size() != Observation.SIZE) {
            return false;
        }

        if (observation.tick() < 0L) {
            return false;
        }

        return observation.isFinite();
    }

    /**
     * Validates the feature vector itself.
     */
    private static void validateValues(Observation observation) {
        for (int i = 0; i < observation.size(); i++) {
            float value = observation.get(i);

            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Non-finite observation value at index "
                                + i + ": " + value
                );
            }
        }
    }

    /**
     * Verifies that the observation ABI hasn't unexpectedly changed.
     *
     * This should be useful during initialization/tests so a network
     * cannot silently receive a different input dimension.
     */
    public static void verifyExpectedSize(int networkInputSize) {
        if (networkInputSize != Observation.SIZE) {
            throw new IllegalArgumentException(
                    "Neural network input size "
                            + networkInputSize
                            + " does not match observation size "
                            + Observation.SIZE
            );
        }
    }

    /**
     * Returns a diagnostic string without throwing.
     */
    public static String diagnose(Observation observation) {
        if (observation == null) {
            return "Observation is null";
        }

        if (observation.size() != Observation.SIZE) {
            return "Observation size mismatch: "
                    + observation.size()
                    + " != "
                    + Observation.SIZE;
        }

        if (observation.tick() < 0L) {
            return "Observation has negative tick: "
                    + observation.tick();
        }

        for (int i = 0; i < observation.size(); i++) {
            float value = observation.get(i);

            if (Float.isNaN(value)) {
                return "Observation contains NaN at index " + i;
            }

            if (Float.isInfinite(value)) {
                return "Observation contains Infinity at index " + i;
            }
        }

        return "OK";
    }
}