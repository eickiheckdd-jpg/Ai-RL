package com.example.newgen6.rl;

/**
 * The initial NEWGEN6 observation ABI: exactly 12 normalized features.
 *
 * Order:
 * 0 relative X
 * 1 relative Y
 * 2 relative Z
 * 3 distance
 * 4 yaw error
 * 5 pitch error
 * 6 self velocity X
 * 7 self velocity Y
 * 8 self velocity Z
 * 9 target velocity X
 * 10 target velocity Y
 * 11 target velocity Z
 */
public final class Observation {
    public static final int SIZE = 12;

    private final float[] features;
    private final boolean targetValid;
    private final double distance;
    private final float yawErrorDegrees;
    private final float pitchErrorDegrees;
    private final String targetName;

    private Observation(
            float[] features,
            boolean targetValid,
            double distance,
            float yawErrorDegrees,
            float pitchErrorDegrees,
            String targetName) {
        if (features.length != SIZE) {
            throw new IllegalArgumentException("Expected " + SIZE + " features, got " + features.length);
        }
        this.features = features;
        this.targetValid = targetValid;
        this.distance = distance;
        this.yawErrorDegrees = yawErrorDegrees;
        this.pitchErrorDegrees = pitchErrorDegrees;
        this.targetName = targetName;
    }

    public static Observation noTarget() {
        return new Observation(new float[SIZE], false, 0.0, 0.0f, 0.0f, "NONE");
    }

    public static Observation of(
            float[] features,
            double distance,
            float yawErrorDegrees,
            float pitchErrorDegrees,
            String targetName) {
        return new Observation(features, true, distance, yawErrorDegrees, pitchErrorDegrees, targetName);
    }

    public float[] features() {
        return features;
    }

    public boolean targetValid() {
        return targetValid;
    }

    public double distance() {
        return distance;
    }

    public float yawErrorDegrees() {
        return yawErrorDegrees;
    }

    public float pitchErrorDegrees() {
        return pitchErrorDegrees;
    }

    public String targetName() {
        return targetName;
    }

    public double angularErrorDegrees() {
        return Math.sqrt(
                (double) yawErrorDegrees * yawErrorDegrees
                        + (double) pitchErrorDegrees * pitchErrorDegrees);
    }
}
