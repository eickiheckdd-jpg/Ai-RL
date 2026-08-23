package com.example.newgen6;

import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;

public class RewardShaper {
    
    // Rewritten for PURE AIM TRAINING
    public float computeReward(PlayerEntity agent, LivingEntity target, float yawDelta, float pitchDelta, float yawErrorNorm, float pitchErrorNorm, boolean targetPresent) {
        if (agent == null || target == null) return 0.0f;

        float reward = 0.0f;

        // 1. Accuracy Reward: the closer to 0 error, the higher the reward.
        // yawErrorNorm and pitchErrorNorm are in [-1, 1] range.
        float totalError = Math.abs(yawErrorNorm) + Math.abs(pitchErrorNorm);
        reward += (1.0f - totalError) * 0.1f; 

        // 2. Raycast Dwell Reward: tiny bonus for actually having the crosshair directly on the hitbox
        if (targetPresent) {
            reward += 0.05f;
        } else {
            // Slight step cost for losing the target entirely
            reward -= 0.01f;
        }

        // 3. Anti-Jitter Penalty: prevent erratic camera shaking to exploit tracking rewards
        float angularVelocity = Math.abs(yawDelta) + Math.abs(pitchDelta);
        if (angularVelocity > 15.0f) {
            reward -= 0.005f * (angularVelocity - 15.0f);
        }

        return Float.isNaN(reward) ? 0.0f : reward;
    }

    public void reset() {
        // Kept for future state tracking (e.g., when adding movement/attacks back)
    }
}
