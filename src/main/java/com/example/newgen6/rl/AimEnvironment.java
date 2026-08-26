package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.AbstractClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Vec3d;

/**
 * Real Minecraft environment for the first aiming/control experiment.
 */
public final class AimEnvironment {
    // These are the action-to-camera scales used by the learned policy.
    // They are not an aiming rule; the network still chooses the action.
    private static final float MAX_YAW_DEGREES_PER_TICK = 6.0f;
    private static final float MAX_PITCH_DEGREES_PER_TICK = 5.0f;

    private Observation previousObservation;

    public Observation observe(MinecraftClient client) {
        if (client.player == null || client.world == null) {
            previousObservation = Observation.noTarget();
            return previousObservation;
        }

        PlayerEntity self = client.player;
        AbstractClientPlayerEntity target = findNearestTarget(client, self);
        if (target == null) {
            previousObservation = Observation.noTarget();
            return previousObservation;
        }

        double dx = target.getX() - self.getX();
        double dy = target.getEyeY() - self.getEyeY();
        double dz = target.getZ() - self.getZ();
        double horizontalDistance = Math.sqrt(dx * dx + dz * dz);
        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        float desiredYaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float desiredPitch = (float) Math.toDegrees(-Math.atan2(dy, horizontalDistance));
        float yawError = wrapDegrees(desiredYaw - self.getYaw());
        float pitchError = clamp(desiredPitch - self.getPitch(), -90.0f, 90.0f);

        Vec3d selfVelocity = self.getVelocity();
        Vec3d targetVelocity = target.getVelocity();

        float[] features = new float[Observation.SIZE];
        features[0] = finite((float) (dx / 16.0));
        features[1] = finite((float) (dy / 8.0));
        features[2] = finite((float) (dz / 16.0));
        features[3] = finite((float) (distance / 24.0));
        features[4] = finite(yawError / 180.0f);
        features[5] = finite(pitchError / 90.0f);
        features[6] = finite((float) selfVelocity.x);
        features[7] = finite((float) selfVelocity.y);
        features[8] = finite((float) selfVelocity.z);
        features[9] = finite((float) targetVelocity.x);
        features[10] = finite((float) targetVelocity.y);
        features[11] = finite((float) targetVelocity.z);

        previousObservation = Observation.of(
                features,
                distance,
                yawError,
                pitchError,
                target.getName().getString());
        return previousObservation;
    }

    public void applyAction(PlayerEntity player, float yawAction, float pitchAction) {
        if (player == null) {
            return;
        }

        float nextYaw = wrapDegrees(player.getYaw() + yawAction * MAX_YAW_DEGREES_PER_TICK);
        float nextPitch = clamp(player.getPitch() - pitchAction * MAX_PITCH_DEGREES_PER_TICK, -90.0f, 90.0f);

        if (Float.isFinite(nextYaw)) {
            player.setYaw(nextYaw);
        }
        if (Float.isFinite(nextPitch)) {
            player.setPitch(nextPitch);
        }
    }

    public float reward(Observation before, Observation after) {
        if (!before.targetValid() || !after.targetValid()) {
            return 0.0f;
        }

        double improvement = before.angularErrorDegrees() - after.angularErrorDegrees();
        return clamp((float) (improvement / 180.0), -1.0f, 1.0f);
    }

    private static AbstractClientPlayerEntity findNearestTarget(MinecraftClient client, PlayerEntity self) {
        AbstractClientPlayerEntity nearest = null;
        double nearestDistanceSquared = Double.POSITIVE_INFINITY;

        for (AbstractClientPlayerEntity candidate : client.world.getPlayers()) {
            if (candidate == self || candidate.isRemoved() || candidate.isDead()) {
                continue;
            }

            double distanceSquared = self.squaredDistanceTo(candidate);
            if (!Double.isFinite(distanceSquared)) {
                continue;
            }

            if (distanceSquared < nearestDistanceSquared) {
                nearestDistanceSquared = distanceSquared;
                nearest = candidate;
            }
        }

        return nearest;
    }

    public static float wrapDegrees(float degrees) {
        float wrapped = degrees % 360.0f;
        if (wrapped >= 180.0f) {
            wrapped -= 360.0f;
        }
        if (wrapped < -180.0f) {
            wrapped += 360.0f;
        }
        return wrapped;
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float finite(float value) {
        return Float.isFinite(value) ? value : 0.0f;
    }
}
