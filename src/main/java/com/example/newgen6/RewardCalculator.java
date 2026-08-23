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
        
        // 1. Base Step Penalty (The "Hurry Up" mechanic)
        // Bleeds a tiny bit of reward every tick so the AI cannot just stand still or hide safely.
        reward -= 0.01f;

        // 2. Health & Damage Mechanics (The Ultimate Goal)
        float currentTargetHealth = target.getHealth();
        float currentPlayerHealth = client.player.getHealth();

        if (previousTargetHealth != -1.0f) {
            float damageDealt = previousTargetHealth - currentTargetHealth;
            if (damageDealt > 0) {
                reward += (damageDealt * 10.0f); // Massive payout for actual progression
            }
        }
        
        if (previousPlayerHealth != -1.0f) {
            float damageTaken = previousPlayerHealth - currentPlayerHealth;
            if (damageTaken > 0) {
                reward -= (damageTaken * 5.0f); // Heavy penalty for taking a hit
            }
        }

        previousTargetHealth = currentTargetHealth;
        previousPlayerHealth = currentPlayerHealth;

        // 3. Anti-Wiggle Distance Reward
        // Only rewards the AI if it pushes closer than it has EVER been this round.
        if (currentDistance < minDistanceAchieved) {
            float distanceClosed = (float) (minDistanceAchieved - currentDistance);
            reward += distanceClosed * 0.5f; 
            minDistanceAchieved = currentDistance;
        }
        
        // 4. Distance-Scaled Aim Reward (Fixes the Staring Exploit)
        // Aiming is only rewarded if the target is within 8 blocks. 
        // The closer the bot gets, the higher the aim reward multiplier.
        if (aimAlignment > 0.85f) {
            if (currentDistance <= 3.0) {
                reward += 0.05f; // Perfect aim inside combat reach
            } else if (currentDistance <= 8.0) {
                // Diminishing returns: drops to 0 reward as distance approaches 8 blocks
                reward += 0.05f * (float)(1.0 - (currentDistance / 8.0));
            }
        }

        // 5. Action Penalties & Payouts
        if (wasInvalid) {
            reward -= 0.5f; // Stop spam-clicking the air or swinging on cooldown
        }
        
        if (wasUnnecessaryJump) {
            // Heavy penalty for jumping if not aiming at the target or if too far away
            reward -= 0.2f; 
        }

        if (wasValidAttack) {
            reward += 1.5f; // Good timing, crosshair actually on target
        }
        
        if (wasCrit) {
            reward += 3.0f; // Jackpot reward for organically discovering jump-crits
        }

        previousDistance = currentDistance;
        return reward;
    }
}
