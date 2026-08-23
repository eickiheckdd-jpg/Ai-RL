package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.MathHelper;

public class PerceptionSystem {

    public static float[] getObservation(MinecraftClient client, LivingEntity target) {
        float[] obs = new float[16];
        ClientPlayerEntity player = client.player;

        if (player == null) return obs;

        // Player states
        obs[0] = player.getHealth() / 20.0f;
        obs[1] = player.getAttackCooldownProgress(0.0f);
        obs[2] = player.isOnGround() ? 1.0f : 0.0f;
        obs[3] = (float) Math.max(-1.0, Math.min(1.0, player.getVelocity().y));

        if (target != null) {
            double dx = target.getX() - player.getX();
            double dy = target.getEyeY() - player.getEyeY();
            double dz = target.getZ() - player.getZ();

            // THE FIX: Distance floor of 0.5 prevents division by zero at point-blank range
            double horizontalDist = Math.max(0.5, Math.sqrt(dx * dx + dz * dz));
            double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            obs[4] = (float) Math.min(1.0, distance / 32.0); // Normalized distance

            // Calculate target angles using stabilized horizontal distance
            double targetYaw = Math.toDegrees(Math.atan2(dz, dx)) - 90.0;
            double targetPitch = Math.toDegrees(-Math.atan2(dy, horizontalDist));

            double yawDiff = MathHelper.wrapDegrees(targetYaw - player.getYaw());
            double pitchDiff = MathHelper.wrapDegrees(targetPitch - player.getPitch());

            float yawDiffNorm = (float) (yawDiff / 180.0);
            float pitchDiffNorm = (float) (pitchDiff / 90.0);

            obs[5] = yawDiffNorm;
            obs[6] = pitchDiffNorm;

            // Total aim error (aimAlignment)
            float totalError = (float) Math.min(1.0, Math.sqrt(yawDiffNorm * yawDiffNorm + pitchDiffNorm * pitchDiffNorm));
            obs[7] = target.getHealth() / 20.0f;
            obs[8] = totalError; // INDEX 8 is read by the TickHandler for alignment
            
            obs[9] = target.hurtTime > 0 ? 1.0f : 0.0f;
        } else {
            obs[8] = 1.0f; // Max error if no target
        }

        return obs;
    }
}
