package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.Vec3d;

public class RewardCalculator {
    private float previousTargetHealth = -1.0f;
    private float previousPlayerHealth = -1.0f;
    private double previousPotential = 0.0;

    private static final double OPTIMAL_DISTANCE = 2.8; // Ideal PvP sword range
    private static final double DISTANCE_SIGMA = 1.5;   // Spread of Gaussian distribution

    public void reset(MinecraftClient client, LivingEntity target) {
        if (target != null) {
            this.previousTargetHealth = target.getHealth();
        }
        if (client.player != null) {
            this.previousPlayerHealth = client.player.getHealth();
            if (target != null) {
                this.previousPotential = calculatePotential(client, target);
            }
        }
    }

    public float calculateReward(MinecraftClient client, LivingEntity target) {
        if (client.player == null || target == null) return 0.0f;
        float reward = 0.0f;

        // --- 1. LETHAL TERMINAL EVENTS ---
        if (target.isDead()) reward += 5.0f;
        if (client.player.isDead()) reward -= 5.0f;

        // --- 2. EXACT HEALTH DELTA SCALING ---
        float currentTargetHealth = target.getHealth();
        if (previousTargetHealth != -1.0f && currentTargetHealth < previousTargetHealth) {
            float damageDealt = previousTargetHealth - currentTargetHealth;
            float cooldownFactor = client.player.getAttackCooldownProgress(0.0f);
            reward += (damageDealt * 0.25f) * (0.5f + 0.5f * cooldownFactor);
        }
        previousTargetHealth = currentTargetHealth;

        float currentPlayerHealth = client.player.getHealth();
        if (previousPlayerHealth != -1.0f && currentPlayerHealth < previousPlayerHealth) {
            float damageReceived = previousPlayerHealth - currentPlayerHealth;
            reward -= (damageReceived * 0.25f);
        }
        previousPlayerHealth = currentPlayerHealth;

        // --- 3. FIXED POTENTIAL-BASED REWARD SHAPING ---
        double currentPotential = calculatePotential(client, target);
        
        // Pure state transition delta (gamma = 1.0 to prevent decay penalties)
        double shapingReward = currentPotential - previousPotential;
        reward += (float) shapingReward;
        previousPotential = currentPotential;

        // --- 4. DIRECT DENSE AIM INCENTIVE (Crucial for Curriculum Phase 1) ---
        // Give a steady positive flow (+0.02 max per tick) for keeping crosshair centered
        double alignment = getRawAlignment(client, target); // [0.0 to 1.0]
        if (alignment > 0.8) { 
            reward += (float) (alignment * 0.02); // Direct positive reward for holding good aim
        }

        return reward;
    }

    /**
     * Calculates state potential phi(s) strictly bounded between 0.0 and 1.0.
     */
    private double calculatePotential(MinecraftClient client, LivingEntity target) {
        double cosineAlignment = getRawAlignment(client, target); // Bounded [0.0, 1.0]

        Vec3d playerEyePos = client.player.getEyePos();
        Vec3d targetCenter = target.getBoundingBox().getCenter();
        double distance = targetCenter.subtract(playerEyePos).length();

        // Gaussian Distance Potential [0.0, 1.0] centered around 2.8 blocks
        double distancePotential = Math.exp(-Math.pow(distance - OPTIMAL_DISTANCE, 2) / (2 * Math.pow(DISTANCE_SIGMA, 2)));

        // Both components are strictly positive [0.0, 1.0]
        return (0.7 * cosineAlignment) + (0.3 * distancePotential);
    }

    /**
     * Helper to compute 3D Cosine Alignment normalized to range [0.0, 1.0].
     * 1.0 = Dead Center Aim, 0.5 = 90 degrees away, 0.0 = Looking directly behind.
     */
    private double getRawAlignment(MinecraftClient client, LivingEntity target) {
        Vec3d playerEyePos = client.player.getEyePos();
        Vec3d targetCenter = target.getBoundingBox().getCenter();
        Vec3d toTargetNormalized = targetCenter.subtract(playerEyePos).normalize();
        Vec3d lookVec = client.player.getRotationVec(1.0f).normalize();

        double dotProduct = lookVec.dotProduct(toTargetNormalized); // [-1.0, 1.0]
        return (dotProduct + 1.0) / 2.0; // Normalized to [0.0, 1.0]
    }
}
