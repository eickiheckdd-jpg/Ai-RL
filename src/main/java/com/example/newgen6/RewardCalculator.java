package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;

public class RewardCalculator {
    private double previousDistance = 999.0;
    private float previousTargetHealth = -1.0f;
    private float previousPlayerHealth = -1.0f;
    
    // Anti-Wiggle Tracking
    private double minDistanceAchieved = 999.0;

    public void reset(MinecraftClient client, LivingEntity target) {
        if (client.player != null && target != null) {
            this.previousDistance = client.player.distanceTo(target);
            this.minDistanceAchieved = this.previousDistance;
            this.previousPlayerHealth = client.player.getHealth();
            this.previousTargetHealth = target.getHealth();
        } else {
            this.previousDistance = 999.0;
            this.minDistanceAchieved = 999.0;
            this.previousPlayerHealth = 20.0f;
            this.previousTargetHealth = 20.0f;
        }
    }

    public float calculateReward(MinecraftClient client, LivingEntity target, 
                                 boolean wasInvalid, boolean wasUnnecessaryJump, 
                                 boolean wasValidAttack, boolean wasCrit, float aimAlignment) {
        if (client.player == null || target == null) return 0.0f;

        float reward = 0.0f;
        double currentDistance = client.player.distanceTo(target);
        
        // 1. Base Step Penalty (Forces the bot to stay active and press forward)
        reward -= 0.01f;

        // 2. Health & Damage Mechanics (The Primary Objective)
        float currentTargetHealth = target.getHealth();
        float currentPlayerHealth = client.player.getHealth();

        if (previousTargetHealth != -1.0f) {
            float damageDealt = previousTargetHealth - currentTargetHealth;
            if (damageDealt > 0) {
                reward += (damageDealt * 10.0f); 
            }
        }
        
        if (previousPlayerHealth != -1.0f) {
            float damageTaken = previousPlayerHealth - currentPlayerHealth;
            if (damageTaken > 0) {
                reward -= (damageTaken * 5.0f); 
            }
        }

        previousTargetHealth = currentTargetHealth;
        previousPlayerHealth = currentPlayerHealth;

        // 3. Anti-Wiggle Distance Reward (Patched: Requires looking at the target)
        // Blocks moonwalking or backing up from farming distance points.
        if (currentDistance < minDistanceAchieved && aimAlignment > 0.3f) {
            float distanceClosed = (float) (minDistanceAchieved - currentDistance);
            reward += distanceClosed * 0.5f; 
            minDistanceAchieved = currentDistance;
        }
        
        // 4. Close-Range Aim & Tracking Reward (Patched: Overcomes step penalty up close)
        if (aimAlignment > 0.80f && currentDistance <= 2.0) {
            reward += 0.05f; // Neutralizes the -0.01 step penalty and rewards close tracking
        } else if (aimAlignment > 0.85f && currentDistance <= 8.0) {
            // Distance-scaled aim reward for mid-range tracking
            reward += 0.05f * (float)(1.0 - (currentDistance / 8.0));
        }

        // 5. Action Penalties & Combat Payouts
        if (wasInvalid) {
            reward -= 0.5f; // Penalty for swinging wildly on cooldown or hitting empty air
        }
        
        if (wasUnnecessaryJump) {
            reward -= 0.2f; // Penalty for random spam-jumping away from target
        }

        if (wasValidAttack) {
            reward += 1.5f; // Reward for properly timed attack with crosshair on target
        }
        
        if (wasCrit) {
            reward += 3.0f; // High reward for landing a critical hit
        }

        previousDistance = currentDistance;
        return reward;
    }
}