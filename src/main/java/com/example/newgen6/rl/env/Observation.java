package com.example.newgen6.rl.env;

import java.util.Arrays;

/**
 * Immutable RL observation.
 *
 * This is the boundary between the Minecraft environment and the
 * reinforcement-learning system.
 *
 * The observation contains no actions and no combat decisions.
 */
public final class Observation {

    public static final int SIZE = ObservationEncoder.OBSERVATION_SIZE;

    private final float[] values;
    private final long tick;
    private final boolean valid;
    private final boolean targetPresent;

    /**
     * Creates an observation from an already encoded feature vector.
     *
     * The input is defensively copied so the neural network cannot
     * accidentally modify the environment's observation.
     */
    public Observation(
            float[] values,
            long tick,
            boolean valid,
            boolean targetPresent) {

        if (values == null) {
            throw new IllegalArgumentException("Observation values cannot be null");
        }

        if (values.length != SIZE) {
            throw new IllegalArgumentException(
                    "Observation size mismatch: got "
                            + values.length
                            + ", expected "
                            + SIZE
            );
        }

        this.values = Arrays.copyOf(values, SIZE);
        this.tick = tick;
        this.valid = valid;
        this.targetPresent = targetPresent;

        validateFinite(this.values);
    }

    /**
     * Creates an Observation from a raw Minecraft snapshot.
     */
    public static Observation fromSnapshot(
            com.example.newgen6.mixin.PvPMixin.Snapshot snapshot,
            long tick) {

        if (snapshot == null) {
            return empty(tick);
        }

        float[] encoded = ObservationEncoder.encode(snapshot);

        return new Observation(
                encoded,
                tick,
                snapshot.valid,
                snapshot.targetPresent
        );
    }

    /**
     * Creates a zero observation.
     *
     * This is useful when Minecraft state is temporarily unavailable,
     * for example during client initialization or world transitions.
     */
    public static Observation empty(long tick) {
        return new Observation(
                ObservationEncoder.emptyObservation(),
                tick,
                false,
                false
        );
    }

    /**
     * Returns a defensive copy of the complete observation.
     */
    public float[] copyValues() {
        return Arrays.copyOf(values, values.length);
    }

    /**
     * Returns a single feature.
     */
    public float get(int index) {
        if (index < 0 || index >= SIZE) {
            throw new IndexOutOfBoundsException(
                    "Observation index " + index
                            + " outside [0, " + (SIZE - 1) + "]"
            );
        }

        return values[index];
    }

    /**
     * Returns the number of observation features.
     */
    public int size() {
        return SIZE;
    }

    /**
     * Minecraft client tick at which this observation was captured.
     */
    public long tick() {
        return tick;
    }

    /**
     * Whether the underlying Minecraft snapshot was valid.
     */
    public boolean isValid() {
        return valid;
    }

    /**
     * Whether a target entity was present when this observation
     * was captured.
     */
    public boolean hasTarget() {
        return targetPresent;
    }

    /**
     * Returns whether every feature is finite.
     */
    public boolean isFinite() {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Creates a copy with a different tick.
     *
     * The feature vector itself remains unchanged.
     */
    public Observation withTick(long newTick) {
        return new Observation(
                values,
                newTick,
                valid,
                targetPresent
        );
    }

    /**
     * Creates a detached copy of this observation.
     */
    public Observation copy() {
        return new Observation(
                values,
                tick,
                valid,
                targetPresent
        );
    }

    private static void validateFinite(float[] values) {
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                throw new IllegalArgumentException(
                        "Observation contains non-finite value at index "
                                + i + ": " + values[i]
                );
            }
        }
    }

    @Override
    public String toString() {
        return "Observation{" +
                "size=" + SIZE +
                ", tick=" + tick +
                ", valid=" + valid +
                ", targetPresent=" + targetPresent +
                '}';
    }
}