package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;

public class RewardCalculator {
    private float lastPlayerHealth = 20.0f;
    private float lastTargetHealth = 20.0f;
    private LivingEntity lastTargetRef = null;

    public static float DAMAGE_REWARD = 2.0f;
    public static float DAMAGE_TAKEN_PENALTY = 2.5f;
    public static float KILL_REWARD = 10.0f;
    public static float DEATH_PENALTY = 10.0f;
    public static float AIM_ALIGNMENT_REWARD = 0.05f;
    public static float INVALID_ACTION_PENALTY = 0.1f;
    public static float UNNECESSARY_JUMP_PENALTY = 0.05f;

    public float calculateReward(MinecraftClient client, LivingEntity target, boolean wasInvalidAction, boolean wasUnnecessaryJump, float aimAlignment) {
        if (client.player == null) return 0.0f;

        float reward = 0.0f;
        float currentPlayerHealth = client.player.getHealth();

        // Player damage taken penalties
        if (currentPlayerHealth < lastPlayerHealth) {
            reward -= (lastPlayerHealth - currentPlayerHealth) * DAMAGE_TAKEN_PENALTY;
        }
        if (client.player.isDead()) {
            reward -= DEATH_PENALTY;
        }

        // Target damage rewards
        if (target != null) {
            if (lastTargetRef != target) {
                lastTargetRef = target;
                lastTargetHealth = target.getHealth();
            }

            float currentTargetHealth = target.getHealth();
            if (currentTargetHealth < lastTargetHealth) {
                reward += (lastTargetHealth - currentTargetHealth) * DAMAGE_REWARD;
            }
            if (target.isDead()) {
                reward += KILL_REWARD;
            }
            lastTargetHealth = currentTargetHealth;

            // SCALED AIM PENALTY: 
            // Multiplying by 4.0f makes off-target states drop into a heavier negative reward (-0.2f max)
            // instead of a weak -0.05f tap, forcing the network to correct its aim faster.
            reward += (0.5f - Math.abs(aimAlignment)) * (AIM_ALIGNMENT_REWARD * 4.0f);
        } else {
            lastTargetRef = null;
            lastTargetHealth = 20.0f;
        }

        if (wasInvalidAction) {
            reward -= INVALID_ACTION_PENALTY;
        }

        if (wasUnnecessaryJump) {
            reward -= UNNECESSARY_JUMP_PENALTY;
        }

        lastPlayerHealth = currentPlayerHealth;
        return reward;
    }

    public void reset(MinecraftClient client, LivingEntity target) {
        this.lastPlayerHealth = client.player != null ? client.player.getHealth() : 20.0f;
        this.lastTargetRef = target;
        this.lastTargetHealth = target != null ? target.getHealth() : 20.0f;
    }
}
