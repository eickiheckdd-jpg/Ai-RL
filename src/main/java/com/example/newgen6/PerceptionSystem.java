package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;

public class PerceptionSystem {
    public static float[] getObservation(MinecraftClient client, LivingEntity target) {
        float[] obs = new float[16];
        PlayerEntity player = client.player;
        if (player == null) return obs;

        obs[0] = player.getHealth() / player.getMaxHealth();
        obs[1] = player.getAttackCooldownProgress(0.0f);
        obs[2] = player.isBlocking() ? 1.0f : 0.0f;
        
        if (target != null) {
            boolean visible = client.crosshairTarget != null && 
                              client.crosshairTarget.getType() == HitResult.Type.ENTITY && 
                              ((EntityHitResult)client.crosshairTarget).getEntity().equals(target);
            
            obs[3] = visible ? 1.0f : 0.0f;
            obs[4] = target.getHealth() / target.getMaxHealth();
            
            double dx = target.getX() - player.getX();
            double dy = target.getEyeY() - player.getEyeY();
            double dz = target.getZ() - player.getZ();
            double distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            obs[5] = (float) Math.min(1.0, distance / 32.0);

            double targetYaw = Math.toDegrees(Math.atan2(dz, dx)) - 90;
            double targetPitch = Math.toDegrees(-Math.atan2(dy, Math.sqrt(dx*dx + dz*dz)));
            
            float yawDiff = (float) Math.max(-1.0, Math.min(1.0, (targetYaw - player.getYaw()) / 180.0));
            float pitchDiff = (float) Math.max(-1.0, Math.min(1.0, (targetPitch - player.getPitch()) / 90.0));
            
            obs[6] = (float) Math.sin(yawDiff * Math.PI);
            obs[7] = (float) Math.cos(yawDiff * Math.PI);
            obs[8] = pitchDiff; // This doubles as aim alignment
        }
        
        return obs;
    }
}
