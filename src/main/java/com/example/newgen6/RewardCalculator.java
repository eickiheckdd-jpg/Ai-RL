package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;

public class RewardCalculator {
    private float lastPlayerHealth = 20.0f;
    private float lastTargetHealth = 20.0f;

    public static float DAMAGE_REWARD = 2.0f;
    public static float DAMAGE_TAKEN_PENALTY = 2.5f;
    public static float KILL_REWARD = 10.0f;
    public static float DEATH_PENALTY = 10.0f;
    public static float AIM_ALIGNMENT_REWARD = 0.05f;
    public static float INVALID_ACTION_PENALTY = 0.1f;

    public float calculateReward(MinecraftClient client, LivingEntity target, boolean wasInvalidAction, float aimAlignment) {
        if (client.player == null) return 0.0f;

        float reward = 0.0f;
        float currentPlayerHealth = client.player.getHealth();

        // Health penalties (Player)
        if (currentPlayerHealth < lastPlayerHealth) {
            reward -= (lastPlayerHealth - currentPlayerHealth) * DAMAGE_TAKEN_PENALTY;
        }
        if (client.player.isDead()) {
            reward -= DEATH_PENALTY;
        }

        // Combat rewards (Target)
        if (target != null) {
            float currentTargetHealth = target.getHealth();
            if (currentTargetHealth < lastTargetHealth) {
                reward += (lastTargetHealth - currentTargetHealth) * DAMAGE_REWARD;
            }
            if (target.isDead()) {
                reward += KILL_REWARD;
            }
            lastTargetHealth = currentTargetHealth;
            
            // SHIFTED GRADIENT: 
            // If aimAlignment is 0.0 (perfect), reward is +0.05
            // If aimAlignment is 1.0 (terrible), reward is -0.05
            reward += (0.5f - Math.abs(aimAlignment)) * (AIM_ALIGNMENT_REWARD * 2.0f);
        } else {
            lastTargetHealth = 20.0f;
        }

        if (wasInvalidAction) reward -= INVALID_ACTION_PENALTY;
        lastPlayerHealth = currentPlayerHealth;

        return reward;
    }

    public void reset(MinecraftClient client, LivingEntity target) {
        this.lastPlayerHealth = client.player != null ? client.player.getHealth() : 20.0f;
        this.lastTargetHealth = target != null ? target.getHealth() : 20.0f;
    }
}
