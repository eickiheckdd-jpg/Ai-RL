package com.example.newgen6.rl.env;

import com.example.newgen6.mixin.PvPMixin;

/**
 * Converts the Minecraft PvP snapshot into the fixed PPO observation ABI.
 *
 * Observation size is exactly 229 floats.
 *
 * IMPORTANT:
 * The first 183 features are the actual PvP features.
 * The remaining 46 are reserved deterministic features so the
 * observation ABI remains stable for the PPO network.
 */
public final class ObservationEncoder {

    public static final int OBSERVATION_SIZE = 229;

    private static final int ENCODED_CORE_SIZE = 183;

    private static final float POSITION_SCALE = 16.0f;
    private static final float DISTANCE_SCALE = 16.0f;
    private static final float HEALTH_SCALE = 20.0f;
    private static final float ANGLE_SCALE = 180.0f;

    private ObservationEncoder() {
    }

    public static float[] encode(PvPMixin.Snapshot s) {

        if (s == null) {
            return emptyObservation();
        }

        float[] out = new float[OBSERVATION_SIZE];

        int i = 0;

        /*
         * =========================================================
         * SELF STATE
         * =========================================================
         */

        out[i++] = finite((float) (s.selfX / POSITION_SCALE));
        out[i++] = finite((float) (s.selfY / POSITION_SCALE));
        out[i++] = finite((float) (s.selfZ / POSITION_SCALE));

        out[i++] = finite((float) s.selfVelocityX);
        out[i++] = finite((float) s.selfVelocityY);
        out[i++] = finite((float) s.selfVelocityZ);

        out[i++] = angleNormalized(s.selfYaw);
        out[i++] = angleNormalized(s.selfPitch);

        out[i++] = clamp01(s.selfHealth / HEALTH_SCALE);
        out[i++] = clamp01(s.selfAbsorption / HEALTH_SCALE);

        out[i++] = bool(s.selfOnGround);
        out[i++] = bool(s.selfSprinting);
        out[i++] = bool(s.selfSneaking);

        out[i++] = clamp01(s.selfAttackCooldown);

        out[i++] = bool(s.valid);

        float selfSpeed = length(
                (float) s.selfVelocityX,
                (float) s.selfVelocityY,
                (float) s.selfVelocityZ
        );

        float horizontalSpeed = length(
                (float) s.selfVelocityX,
                (float) s.selfVelocityZ
        );

        out[i++] = finite(selfSpeed);
        out[i++] = finite(horizontalSpeed);

        out[i++] = finite((float) s.selfVelocityX);
        out[i++] = finite((float) s.selfVelocityY);
        out[i++] = finite((float) s.selfVelocityZ);

        out[i++] = finite(square((float) s.selfVelocityX));
        out[i++] = finite(square((float) s.selfVelocityY));
        out[i++] = finite(square((float) s.selfVelocityZ));
        out[i++] = finite(square(horizontalSpeed));

        out[i++] = finite(sinDegrees(s.selfYaw));
        out[i++] = finite(cosDegrees(s.selfYaw));
        out[i++] = finite(sinDegrees(s.selfPitch));
        out[i++] = finite(cosDegrees(s.selfPitch));

        out[i++] = finite(sinDegrees(s.selfYaw * 0.5f));
        out[i++] = finite(cosDegrees(s.selfYaw * 0.5f));
        out[i++] = finite(sinDegrees(s.selfPitch * 0.5f));
        out[i++] = finite(cosDegrees(s.selfPitch * 0.5f));

        /*
         * =========================================================
         * TARGET STATE
         * =========================================================
         */

        out[i++] = bool(s.targetPresent);

        out[i++] = finite(
                (float) (s.targetRelativeX / POSITION_SCALE)
        );

        out[i++] = finite(
                (float) (s.targetRelativeY / POSITION_SCALE)
        );

        out[i++] = finite(
                (float) (s.targetRelativeZ / POSITION_SCALE)
        );

        out[i++] = finite((float) s.targetVelocityX);
        out[i++] = finite((float) s.targetVelocityY);
        out[i++] = finite((float) s.targetVelocityZ);

        out[i++] = clamp01(s.targetHealth / HEALTH_SCALE);
        out[i++] = clamp01(s.targetAbsorption / HEALTH_SCALE);

        out[i++] = angleNormalized(s.targetYaw);
        out[i++] = angleNormalized(s.targetPitch);

        out[i++] = bool(s.targetOnGround);
        out[i++] = bool(s.targetSprinting);

        out[i++] = finite(
                (float) (s.targetDistance / DISTANCE_SCALE)
        );

        float targetSpeed = length(
                (float) s.targetVelocityX,
                (float) s.targetVelocityY,
                (float) s.targetVelocityZ
        );

        float targetHorizontalSpeed = length(
                (float) s.targetVelocityX,
                (float) s.targetVelocityZ
        );

        out[i++] = finite(targetSpeed);
        out[i++] = finite(targetHorizontalSpeed);

        out[i++] = finite((float) s.targetVelocityX);
        out[i++] = finite((float) s.targetVelocityY);
        out[i++] = finite((float) s.targetVelocityZ);

        out[i++] = finite(square((float) s.targetRelativeX));
        out[i++] = finite(square((float) s.targetRelativeY));
        out[i++] = finite(square((float) s.targetRelativeZ));

        /*
         * =========================================================
         * RELATIVE GEOMETRY
         * =========================================================
         */

        float rx = (float) s.targetRelativeX;
        float ry = (float) s.targetRelativeY;
        float rz = (float) s.targetRelativeZ;

        float horizontalLen = length(rx, rz);
        float fullLen = length(rx, ry, rz);

        float dirX =
                horizontalLen > 1e-6f
                        ? rx / horizontalLen
                        : 0.0f;

        float dirZ =
                horizontalLen > 1e-6f
                        ? rz / horizontalLen
                        : 0.0f;

        float dirY =
                fullLen > 1e-6f
                        ? ry / fullLen
                        : 0.0f;

        float fullDirX =
                fullLen > 1e-6f
                        ? rx / fullLen
                        : 0.0f;

        float fullDirZ =
                fullLen > 1e-6f
                        ? rz / fullLen
                        : 0.0f;

        float horizontalDistance =
                length(rx, rz);

        float relativeMagnitude =
                length(rx, ry, rz);

        out[i++] = finite(
                horizontalDistance / DISTANCE_SCALE
        );

        out[i++] = finite(
                relativeMagnitude / DISTANCE_SCALE
        );

        out[i++] = finite(dirX);
        out[i++] = finite(dirY);
        out[i++] = finite(dirZ);

        out[i++] = finite(fullDirX);
        out[i++] = finite(dirY);
        out[i++] = finite(fullDirZ);

        /*
         * =========================================================
         * RELATIVE VELOCITY
         * =========================================================
         */

        float tvx = (float) s.targetVelocityX;
        float tvy = (float) s.targetVelocityY;
        float tvz = (float) s.targetVelocityZ;

        float rvx =
                tvx - (float) s.selfVelocityX;

        float rvy =
                tvy - (float) s.selfVelocityY;

        float rvz =
                tvz - (float) s.selfVelocityZ;

        out[i++] = finite(rvx);
        out[i++] = finite(rvy);
        out[i++] = finite(rvz);

        out[i++] = finite(length(rvx, rvy, rvz));
        out[i++] = finite(length(rvx, rvz));

        out[i++] = finite(
                dirX * rvx + dirZ * rvz
        );

        out[i++] = finite(
                fullDirX * rvx
                        + dirY * rvy
                        + fullDirZ * rvz
        );

        /*
         * =========================================================
         * ROTATION
         * =========================================================
         */

        float yawDelta =
                wrapDegrees(
                        s.targetYaw - s.selfYaw
                );

        float pitchDelta =
                wrapDegrees(
                        s.targetPitch - s.selfPitch
                );

        out[i++] = angleNormalized(yawDelta);
        out[i++] = angleNormalized(pitchDelta);

        out[i++] = finite(sinDegrees(yawDelta));
        out[i++] = finite(cosDegrees(yawDelta));

        out[i++] = finite(sinDegrees(pitchDelta));
        out[i++] = finite(cosDegrees(pitchDelta));

        out[i++] = finite(
                sinDegrees(yawDelta * 0.5f)
        );

        out[i++] = finite(
                cosDegrees(yawDelta * 0.5f)
        );

        out[i++] = finite(
                sinDegrees(pitchDelta * 0.5f)
        );

        out[i++] = finite(
                cosDegrees(pitchDelta * 0.5f)
        );

        /*
         * =========================================================
         * AIM GEOMETRY
         * =========================================================
         */

        float horizontalAim =
                horizontalAngle(rx, rz);

        float verticalAim =
                verticalAngle(rx, ry, rz);

        float aimError =
                wrapDegrees(
                        horizontalAim - s.selfYaw
                );

        float pitchAimError =
                wrapDegrees(
                        verticalAim - s.selfPitch
                );

        out[i++] = angleNormalized(horizontalAim);
        out[i++] = finite(sinDegrees(horizontalAim));
        out[i++] = finite(cosDegrees(horizontalAim));

        out[i++] = angleNormalized(verticalAim);
        out[i++] = finite(sinDegrees(verticalAim));
        out[i++] = finite(cosDegrees(verticalAim));

        out[i++] = angleNormalized(aimError);
        out[i++] = finite(sinDegrees(aimError));
        out[i++] = finite(cosDegrees(aimError));

        out[i++] = angleNormalized(pitchAimError);
        out[i++] = finite(sinDegrees(pitchAimError));
        out[i++] = finite(cosDegrees(pitchAimError));

        out[i++] = finite(sinDegrees(s.targetYaw));
        out[i++] = finite(cosDegrees(s.targetYaw));

        out[i++] = finite(sinDegrees(s.targetPitch));
        out[i++] = finite(cosDegrees(s.targetPitch));

        /*
         * =========================================================
         * DISTANCE FEATURES
         * =========================================================
         */

        float distance =
                finiteNonNegative(
                        (float) s.targetDistance
                );

        out[i++] = finite(distance);
        out[i++] = finite(square(distance));
        out[i++] = finite(sqrtSafe(distance));

        out[i++] = finite(
                1.0f / (1.0f + distance)
        );

        out[i++] = finite(
                1.0f / (1.0f + square(distance))
        );

        out[i++] = finite(expDecay(distance, 1));
        out[i++] = finite(expDecay(distance, 2));
        out[i++] = finite(expDecay(distance, 4));
        out[i++] = finite(expDecay(distance, 8));

        out[i++] = step(distance, 1);
        out[i++] = step(distance, 2);
        out[i++] = step(distance, 3);
        out[i++] = step(distance, 4);
        out[i++] = step(distance, 5);
        out[i++] = step(distance, 6);
        out[i++] = step(distance, 8);
        out[i++] = step(distance, 10);

        out[i++] = finite(
                Math.min(distance / 4.0f, 1.0f)
        );

        out[i++] = finite(
                Math.min(distance / 8.0f, 1.0f)
        );

        out[i++] = finite(
                Math.min(distance / 16.0f, 1.0f)
        );

        out[i++] = finite(
                Math.min(distance / 32.0f, 1.0f)
        );

        out[i++] = finite(
                horizontalDistance / 16.0f
        );

        out[i++] = finite(
                Math.abs(ry) / 8.0f
        );

        /*
         * =========================================================
         * COMBAT RELATIONSHIPS
         * =========================================================
         */

        float healthDifference =
                (s.selfHealth - s.targetHealth)
                        / HEALTH_SCALE;

        float absorptionDifference =
                (s.selfAbsorption - s.targetAbsorption)
                        / HEALTH_SCALE;

        out[i++] = finite(healthDifference);
        out[i++] = finite(absorptionDifference);

        out[i++] = clamp01(
                s.selfHealth / HEALTH_SCALE
        );

        out[i++] = clamp01(
                s.targetHealth / HEALTH_SCALE
        );

        out[i++] = clamp01(
                s.selfAbsorption / HEALTH_SCALE
        );

        out[i++] = clamp01(
                s.targetAbsorption / HEALTH_SCALE
        );

        out[i++] = clamp01(
                s.selfAttackCooldown
        );

        float proximity =
                1.0f / (1.0f + distance);

        out[i++] = finite(proximity);

        out[i++] = finite(
                proximity
                        * s.selfHealth
                        / HEALTH_SCALE
        );

        out[i++] = finite(
                proximity
                        * s.targetHealth
                        / HEALTH_SCALE
        );

        float closingSpeed =
                -(rvx * fullDirX
                        + rvy * dirY
                        + rvz * fullDirZ);

        float horizontalClosingSpeed =
                -(rvx * dirX
                        + rvz * dirZ);

        out[i++] = finite(closingSpeed);
        out[i++] = finite(Math.abs(closingSpeed));
        out[i++] = closingSpeed > 0.0f ? 1.0f : 0.0f;

        out[i++] = finite(horizontalClosingSpeed);
        out[i++] = finite(Math.abs(horizontalClosingSpeed));
        out[i++] =
                horizontalClosingSpeed > 0.0f
                        ? 1.0f
                        : 0.0f;

        out[i++] = finite(ry);
        out[i++] = finite(Math.abs(ry));

        out[i++] = ry > 0.0f ? 1.0f : 0.0f;
        out[i++] = ry < 0.0f ? 1.0f : 0.0f;

        out[i++] =
                bool(s.targetPresent && s.targetOnGround);

        out[i++] =
                bool(s.targetPresent && s.targetSprinting);

        out[i++] =
                bool(s.targetPresent && s.selfOnGround);

        out[i++] =
                bool(s.targetPresent && s.selfSprinting);

        /*
         * =========================================================
         * LOCAL MOVEMENT
         * =========================================================
         */

        float selfVX =
                (float) s.selfVelocityX;

        float selfVY =
                (float) s.selfVelocityY;

        float selfVZ =
                (float) s.selfVelocityZ;

        float yawRad =
                (float) Math.toRadians(s.selfYaw);

        float sinYaw =
                (float) Math.sin(yawRad);

        float cosYaw =
                (float) Math.cos(yawRad);

        float localForward =
                -selfVX * sinYaw
                        + selfVZ * cosYaw;

        float localStrafe =
                selfVX * cosYaw
                        + selfVZ * sinYaw;

        out[i++] = finite(localForward);
        out[i++] = finite(localStrafe);
        out[i++] = finite(selfVY);

        out[i++] = finite(square(localForward));
        out[i++] = finite(square(localStrafe));
        out[i++] = finite(square(selfVY));

        float localTargetForward =
                -rx * sinYaw
                        + rz * cosYaw;

        float localTargetStrafe =
                rx * cosYaw
                        + rz * sinYaw;

        out[i++] = finite(
                localTargetForward / POSITION_SCALE
        );

        out[i++] = finite(
                localTargetStrafe / POSITION_SCALE
        );

        out[i++] = finite(
                localTargetForward
                        / Math.max(1.0f, distance)
        );

        out[i++] = finite(
                localTargetStrafe
                        / Math.max(1.0f, distance)
        );

        float localTargetVelocityForward =
                -tvx * sinYaw
                        + tvz * cosYaw;

        float localTargetVelocityStrafe =
                tvx * cosYaw
                        + tvz * sinYaw;

        out[i++] = finite(
                localTargetVelocityForward
        );

        out[i++] = finite(
                localTargetVelocityStrafe
        );

        out[i++] = finite(tvy);

        float localRelativeVelocityForward =
                -rvx * sinYaw
                        + rvz * cosYaw;

        float localRelativeVelocityStrafe =
                rvx * cosYaw
                        + rvz * sinYaw;

        out[i++] = finite(
                localRelativeVelocityForward
        );

        out[i++] = finite(
                localRelativeVelocityStrafe
        );

        out[i++] = finite(rvy);

        out[i++] = finite(
                square(localRelativeVelocityForward)
        );

        out[i++] = finite(
                square(localRelativeVelocityStrafe)
        );

        out[i++] = finite(square(rvy));

        /*
         * =========================================================
         * FINAL STATE FEATURES
         * =========================================================
         */

        out[i++] = bool(s.valid);
        out[i++] = bool(s.targetPresent);
        out[i++] = bool(s.selfOnGround);
        out[i++] = bool(s.selfSprinting);
        out[i++] = bool(s.selfSneaking);
        out[i++] = bool(s.targetOnGround);
        out[i++] = bool(s.targetSprinting);

        out[i++] = finite(targetSpeed);
        out[i++] = finite(horizontalSpeed);
        out[i++] = finite(distance);
        out[i++] = finite(relativeMagnitude);

        out[i++] = finite(
                yawDelta / ANGLE_SCALE
        );

        out[i++] = finite(
                pitchDelta / ANGLE_SCALE
        );

        out[i++] = finite(
                aimError / ANGLE_SCALE
        );

        out[i++] = finite(
                pitchAimError / ANGLE_SCALE
        );

        out[i++] = finite(healthDifference);
        out[i++] = finite(absorptionDifference);
        out[i++] = finite(closingSpeed);
        out[i++] = finite(horizontalClosingSpeed);

        out[i++] = finite(
                clamp01(
                        s.selfHealth / HEALTH_SCALE
                )
        );

        out[i++] = finite(
                clamp01(
                        s.targetHealth / HEALTH_SCALE
                )
        );

        out[i++] = finite(proximity);

        /*
         * =========================================================
         * ABI VALIDATION
         * =========================================================
         *
         * The existing feature construction produces 183 features.
         * Keep the PPO input ABI at 229 by explicitly zero-filling
         * the remaining 46 reserved slots.
         *
         * These are intentionally deterministic and contain no
         * random values or hidden policy information.
         */

        if (i != ENCODED_CORE_SIZE) {
            throw new IllegalStateException(
                    "Internal observation encoder error: encoded "
                            + i
                            + " core features, expected "
                            + ENCODED_CORE_SIZE
            );
        }

        while (i < OBSERVATION_SIZE) {
            out[i++] = 0.0f;
        }

        if (i != OBSERVATION_SIZE) {
            throw new IllegalStateException(
                    "Observation ABI mismatch: encoded "
                            + i
                            + " features, expected "
                            + OBSERVATION_SIZE
            );
        }

        sanitizeInPlace(out);

        return out;
    }

    public static float[] emptyObservation() {
        return new float[OBSERVATION_SIZE];
    }

    private static float finite(float v) {
        return Float.isFinite(v) ? v : 0.0f;
    }

    private static float finiteNonNegative(float v) {
        return !Float.isFinite(v) || v < 0.0f
                ? 0.0f
                : v;
    }

    private static float clamp01(float v) {
        if (!Float.isFinite(v)) {
            return 0.0f;
        }

        return Math.max(
                0.0f,
                Math.min(1.0f, v)
        );
    }

    private static float bool(boolean v) {
        return v ? 1.0f : 0.0f;
    }

    private static float square(float v) {
        if (!Float.isFinite(v)) {
            return 0.0f;
        }

        return v * v;
    }

    private static float sqrtSafe(float v) {
        if (!Float.isFinite(v) || v <= 0.0f) {
            return 0.0f;
        }

        return (float) Math.sqrt(v);
    }

    private static float expDecay(
            float distance,
            float scale) {

        if (!Float.isFinite(distance)
                || distance < 0.0f) {
            return 0.0f;
        }

        return (float) Math.exp(
                -distance
                        / Math.max(scale, 1e-6f)
        );
    }

    private static float step(
            float value,
            float threshold) {

        return value <= threshold
                ? 1.0f
                : 0.0f;
    }

    private static float length(
            float x,
            float y,
            float z) {

        return (float) Math.sqrt(
                x * x
                        + y * y
                        + z * z
        );
    }

    private static float length(
            float x,
            float z) {

        return (float) Math.sqrt(
                x * x
                        + z * z
        );
    }

    private static float wrapDegrees(
            float degrees) {

        float w = degrees % 360.0f;

        if (w >= 180.0f) {
            w -= 360.0f;
        }

        if (w < -180.0f) {
            w += 360.0f;
        }

        return w;
    }

    private static float angleNormalized(
            float degrees) {

        return wrapDegrees(degrees) / 180.0f;
    }

    private static float sinDegrees(
            float degrees) {

        return (float) Math.sin(
                Math.toRadians(degrees)
        );
    }

    private static float cosDegrees(
            float degrees) {

        return (float) Math.cos(
                Math.toRadians(degrees)
        );
    }

    private static float horizontalAngle(
            float x,
            float z) {

        if (Math.abs(x) < 1e-7f
                && Math.abs(z) < 1e-7f) {
            return 0.0f;
        }

        return (float) Math.toDegrees(
                Math.atan2(-x, z)
        );
    }

    private static float verticalAngle(
            float x,
            float y,
            float z) {

        float horizontal =
                length(x, z);

        if (horizontal < 1e-7f
                && Math.abs(y) < 1e-7f) {
            return 0.0f;
        }

        return (float) Math.toDegrees(
                Math.atan2(-y, horizontal)
        );
    }

    private static void sanitizeInPlace(
            float[] values) {

        for (int i = 0;
             i < values.length;
             i++) {

            if (!Float.isFinite(values[i])) {
                values[i] = 0.0f;
            }
        }
    }
}