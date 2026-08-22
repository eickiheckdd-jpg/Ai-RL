package com.example.newgen6.rl;

/**
 * Perception snapshot for a single client tick.
 * Base features (10) + LIDAR cone rays (default 8) = 18 total inputs.
 * FEATURE_COUNT must match the actor/critic MLP input layer size.
 */
public record BotState(
        double relX,            // vector to target, normalized by ~16 blocks
        double relY,
        double relZ,
        double selfHealth,      // 0.0 - 1.0
        double enemyHealth,     // 0.0 - 1.0 (0 if no target locked)
        double selfVelX,
        double selfVelZ,
        double hasLineOfSight,  // 1.0 / 0.0
        double attackCooldown,  // 0.0 (ready) - 1.0 (just swung)
        double onGround,        // 1.0 / 0.0
        double[] lidarDistances // normalized 0.0 (touching) - 1.0 (no hit within range), cone around look vector
) {
    public static final int LIDAR_RAYS = 8;
    public static final int BASE_FEATURES = 10;
    public static final int FEATURE_COUNT = BASE_FEATURES + LIDAR_RAYS;

    public double[] toVector() {
        double[] v = new double[FEATURE_COUNT];
        v[0] = relX; v[1] = relY; v[2] = relZ;
        v[3] = selfHealth; v[4] = enemyHealth;
        v[5] = selfVelX; v[6] = selfVelZ;
        v[7] = hasLineOfSight; v[8] = attackCooldown; v[9] = onGround;
        System.arraycopy(lidarDistances, 0, v, BASE_FEATURES, Math.min(lidarDistances.length, LIDAR_RAYS));
        return v;
    }
}
