package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.MathHelper;

public class RewardCalculator {
    private float previousTargetHealth = -1.0f;
    private float previousPlayerHealth = -1.0f;
    private double previousDistance = -1.0;

    public void reset(MinecraftClient client, LivingEntity target) {
        if (target != null) {
            this.previousTargetHealth = target.getHealth();
        }
        if (client.player != null) {
            this.previousPlayerHealth = client.player.getHealth();
            if (target != null) {
                this.previousDistance = client.player.distanceTo(target);
            }
        }
    }

    public float calculateReward(MinecraftClient client, LivingEntity target) {
        if (client.player == null || target == null) return 0.0f;
        float reward = 0.0f;

        // --- 1. LETHAL EVENTS (Major Rewards/Penalties) ---
        if (target.isDead()) reward += 1.0f;
        if (client.player.isDead()) reward -= 1.0f;

        // --- 2. DAMAGE EVENTS (Moderate Rewards/Penalties) ---
        float currentTargetHealth = target.getHealth();
        if (previousTargetHealth != -1.0f && currentTargetHealth < previousTargetHealth) {
            reward += 0.2f; // Good job, you hurt the player
        }
        previousTargetHealth = currentTargetHealth;

        float currentPlayerHealth = client.player.getHealth();
        if (previousPlayerHealth != -1.0f && currentPlayerHealth < previousPlayerHealth) {
            reward -= 0.2f; // Bad job, you got hit
        }
        previousPlayerHealth = currentPlayerHealth;

        // --- 3. SHAPING EVENTS (Tiny breadcrumbs to guide behavior) ---
        
        // A. Distance Tracking
        double currentDistance = client.player.distanceTo(target);
        if (previousDistance != -1.0) {
            if (currentDistance < previousDistance && currentDistance > 2.0) {
                reward += 0.01f; // Moved closer
            } else if (currentDistance > previousDistance && currentDistance < 10.0) {
                reward -= 0.01f; // Player is getting away
            }
        }
        previousDistance = currentDistance;

        // B. Aim Tracking
        float targetYaw = getYawTo(client.player, target);
        float yawDiff = Math.abs(MathHelper.wrapDegrees(client.player.getYaw() - targetYaw));
        
        if (yawDiff < 15.0f) {
            reward += 0.005f; // Looking right at the player
        } else if (yawDiff > 45.0f) {
            reward -= 0.005f; // Looking away from the player
        }

        return reward;
    }

    // Helper method to calculate where the bot should be looking
    private float getYawTo(Entity player, Entity target) {
        double dx = target.getX() - player.getX();
        double dz = target.getZ() - player.getZ();
        return (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0);
    }
}
