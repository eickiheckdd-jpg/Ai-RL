package com.example.newgen6;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;

public class RewardCalculator {
    private float previousTargetHealth = -1.0f;

    public void reset(MinecraftClient client, LivingEntity target) {
        if (target != null) this.previousTargetHealth = target.getHealth();
    }

    public float calculateReward(MinecraftClient client, LivingEntity target) {
        if (client.player == null || target == null) return 0.0f;
        float reward = 0.0f;

        if (target.isDead()) reward += 1.0f;
        if (client.player.isDead()) reward -= 1.0f;

        float currentHealth = target.getHealth();
        if (previousTargetHealth != -1.0f && currentHealth < previousTargetHealth) {
            reward += 0.1f; 
        }
        previousTargetHealth = currentHealth;

        return reward;
    }
}
