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
    private static final double GAMMA = 0.99;            // Discount factor for potential shaping

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
            
            // Reward damage scaled by weapon cooldown readiness
            reward += (damageDealt * 0.25f) * (0.5f + 0.5f * cooldownFactor);
        }
        previousTargetHealth = currentTargetHealth;

        float currentPlayerHealth = client.player.getHealth();
        if (previousPlayerHealth != -1.0f && currentPlayerHealth < previousPlayerHealth) {
            float damageReceived = previousPlayerHealth - currentPlayerHealth;
            reward -= (damageReceived * 0.25f);
        }
        previousPlayerHealth = currentPlayerHealth;

        // --- 3. POTENTIAL-BASED REWARD SHAPING (PBRS) ---
        double currentPotential = calculatePotential(client, target);
        double shapingReward = (GAMMA * currentPotential) - previousPotential;
        reward += (float) shapingReward;
        previousPotential = currentPotential;

        return reward;
    }

    /**
     * Calculates state potential phi(s) based on 3D Aim Cosine Alignment and Gaussian Distance.
     */
    private double calculatePotential(MinecraftClient client, LivingEntity target) {
        Vec3d playerEyePos = client.player.getEyePos();
        Vec3d targetCenter = target.getBoundingBox().getCenter();
        
        // Direction vector to target
        Vec3d toTarget = targetCenter.subtract(playerEyePos);
        double distance = toTarget.length();
        Vec3d toTargetNormalized = toTarget.normalize();

        // Player look vector
        Vec3d lookVec = client.player.getRotationVec(1.0f).normalize();

        // Cosine Alignment: dot product ranges from -1.0 (looking away) to 1.0 (dead center)
        double cosineAlignment = lookVec.dotProduct(toTargetNormalized);

        // Gaussian Distance Potential centered around OPTIMAL_DISTANCE (2.8 blocks)
        double distancePotential = Math.exp(-Math.pow(distance - OPTIMAL_DISTANCE, 2) / (2 * Math.pow(DISTANCE_SIGMA, 2)));

        // Combine continuous orientation and positioning into potential phi(s)
        return (0.6 * cosineAlignment) + (0.4 * distancePotential);
    }
}