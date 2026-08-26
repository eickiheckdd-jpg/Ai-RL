package com.example.newgen6.rl.env;

import com.example.newgen6.mixin.PvPMixin;

import java.util.Arrays;

/**
 * Immutable RL observation.
 *
 * This is the boundary between the Minecraft environment and PPO.
 */
public final class Observation {

    public static final int SIZE =
            ObservationEncoder.OBSERVATION_SIZE;

    private final float[] values;
    private final long tick;
    private final boolean valid;
    private final boolean targetPresent;

    public Observation(
            float[] values,
            long tick,
            boolean valid,
            boolean targetPresent) {

        if (values == null) {
            throw new IllegalArgumentException(
                    "Observation values cannot be null"
            );
        }

        if (values.length != SIZE) {
            throw new IllegalArgumentException(
                    "Observation size mismatch: got "
                            + values.length
                            + ", expected "
                            + SIZE
            );
        }

        this.values =
                Arrays.copyOf(values, SIZE);

        this.tick = tick;
        this.valid = valid;
        this.targetPresent =
                targetPresent;

        validateFinite(this.values);
    }

    public static Observation fromSnapshot(
            PvPMixin.Snapshot snapshot,
            long tick) {

        if (snapshot == null) {
            return empty(tick);
        }

        float[] encoded =
                ObservationEncoder.encode(snapshot);

        return new Observation(
                encoded,
                tick,
                snapshot.valid,
                snapshot.targetPresent
        );
    }

    public static Observation empty(long tick) {
        return new Observation(
                ObservationEncoder.emptyObservation(),
                tick,
                false,
                false
        );
    }

    public float[] copyValues() {
        return Arrays.copyOf(
                values,
                values.length
        );
    }

    public float get(int index) {

        if (index < 0 || index >= SIZE) {
            throw new IndexOutOfBoundsException(
                    "Observation index "
                            + index
                            + " outside [0, "
                            + (SIZE - 1)
                            + "]"
            );
        }

        return values[index];
    }

    public int size() {
        return SIZE;
    }

    public long tick() {
        return tick;
    }

    public boolean isValid() {
        return valid;
    }

    public boolean hasTarget() {
        return targetPresent;
    }

    public boolean isFinite() {

        for (float value : values) {
            if (!Float.isFinite(value)) {
                return false;
            }
        }

        return true;
    }

    public Observation withTick(
            long newTick) {

        return new Observation(
                values,
                newTick,
                valid,
                targetPresent
        );
    }

    public Observation copy() {

        return new Observation(
                values,
                tick,
                valid,
                targetPresent
        );
    }

    private static void validateFinite(
            float[] values) {

        for (int i = 0;
             i < values.length;
             i++) {

            if (!Float.isFinite(values[i])) {
                throw new IllegalArgumentException(
                        "Observation contains "
                                + "non-finite value at index "
                                + i
                                + ": "
                                + values[i]
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
                ", targetPresent=" +
                targetPresent +
                '}';
    }
}