package com.example.newgen6;

import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.MathHelper;

public class CombatActionExecutor {
    public static void execute(PlayerEntity agent, PPOAgent.InferenceResult action) {
        if (Float.isNaN(action.yawDelta()) || Float.isNaN(action.pitchDelta())) return;
        
        float newYaw = agent.getYaw() + action.yawDelta();
        float newPitch = MathHelper.clamp(agent.getPitch() + action.pitchDelta(), -90.0f, 90.0f);
        
        agent.setYaw(newYaw);
        agent.setPitch(newPitch);
    }
} 