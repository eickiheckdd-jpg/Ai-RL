package com.example.newgen6.rl.env;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.Vec3d;

public class RewardCalculator {

    private float lastSelfHealth = -1f;
    private float lastTargetHealth = -1f;

    public float rewardHitDealt = 1.0f;
    public float rewardDamageTaken = -1.0f;
    public float rewardTargetKilled = 5.0f;
    public float rewardSelfDied = -5.0f;
    public float rewardTimeStep = -0.001f;
    public float rewardAiming = 0.02f;

    public float step(ClientPlayerEntity self, LivingEntity target) {
        float selfHealth = self.getHealth();
        float targetHealth = target != null ? target.getHealth() : 0f;
        boolean targetAlive = target != null && target.isAlive();
        boolean selfAlive = self.isAlive();

        float reward = rewardTimeStep;

        if (lastSelfHealth >= 0f && selfHealth < lastSelfHealth) {
            reward += rewardDamageTaken;
        }
        if (lastTargetHealth >= 0f && targetHealth < lastTargetHealth) {
            reward += rewardHitDealt;
        }
        if (lastTargetHealth != 0f && !targetAlive) {
            reward += rewardTargetKilled;
        }
        if (lastSelfHealth != 0f && !selfAlive) {
            reward += rewardSelfDied;
        }

        if (targetAlive) {
            Vec3d rel = target.getPosition().subtract(self.getPosition());
            double desiredYaw = Math.toDegrees(Math.atan2(-rel.x, rel.z));
            float yawDiff = Math.abs(wrapDegrees(self.getYaw() - (float) desiredYaw));
            
            if (yawDiff < 15.0f) {
                reward += rewardAiming;
            }
        }

        lastSelfHealth = selfHealth;
        lastTargetHealth = targetHealth;
        return reward;
    }

    public void reset() {
        lastSelfHealth = -1f;
        lastTargetHealth = -1f;
    }
    
    private static float wrapDegrees(float deg) {
        float d = deg % 360f;
        if (d >= 180f) d -= 360f;
        if (d < -180f) d += 360f;
        return d;
    }
}
