package com.example.newgen6.rl;

/**
 * Discrete aim buckets matching APEX UI: 19 yaw × 17 pitch.
 * Bucket centers map to camera deltas in degrees.
 */
public final class AimBuckets {
    private AimBuckets() {}

    /** Map yaw bucket [0,18] → delta degrees in [-MAX, +MAX]. */
    public static float yawDeltaDeg(int bucket) {
        int b = MathUtil.clamp(bucket, 0, RLConstants.YAW_BUCKETS - 1);
        float t = b / (float) (RLConstants.YAW_BUCKETS - 1); // 0..1
        return (t * 2f - 1f) * RLConstants.MAX_YAW_DELTA_DEG;
    }

    public static float pitchDeltaDeg(int bucket) {
        int b = MathUtil.clamp(bucket, 0, RLConstants.PITCH_BUCKETS - 1);
        float t = b / (float) (RLConstants.PITCH_BUCKETS - 1);
        return (t * 2f - 1f) * RLConstants.MAX_PITCH_DELTA_DEG;
    }

    /** Inverse: continuous delta → nearest bucket (for debugging). */
    public static int yawBucketFromDelta(float deltaDeg) {
        float t = (deltaDeg / RLConstants.MAX_YAW_DELTA_DEG) * 0.5f + 0.5f;
        int b = Math.round(t * (RLConstants.YAW_BUCKETS - 1));
        return MathUtil.clamp(b, 0, RLConstants.YAW_BUCKETS - 1);
    }

    public static int pitchBucketFromDelta(float deltaDeg) {
        float t = (deltaDeg / RLConstants.MAX_PITCH_DELTA_DEG) * 0.5f + 0.5f;
        int b = Math.round(t * (RLConstants.PITCH_BUCKETS - 1));
        return MathUtil.clamp(b, 0, RLConstants.PITCH_BUCKETS - 1);
    }
}
