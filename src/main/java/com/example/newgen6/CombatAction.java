package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;

public enum CombatAction {
    NO_OP,
    AIM_LEFT_SMALL, AIM_LEFT_MEDIUM,
    AIM_RIGHT_SMALL, AIM_RIGHT_MEDIUM,
    AIM_UP_SMALL, AIM_UP_MEDIUM,
    AIM_DOWN_SMALL, AIM_DOWN_MEDIUM,
    
    // New Attack Types
    ATTACK_SPAM,        // Swings immediately, ignoring cooldown
    ATTACK_TIMED_SWEEP, // Only swings if cooldown is high enough for full damage/sweep
    ATTACK_SPRINT,      // Forces sprinting to deal extra knockback
    JUMP,               // Required to set up a Critical Hit on the way down
    
    BLOCK;

    public void execute(MinecraftClient client) {
        if (client.player == null) return;
        
        switch (this) {
            case AIM_LEFT_SMALL:  client.player.setYaw(client.player.getYaw() - 2.0f); break;
            case AIM_LEFT_MEDIUM: client.player.setYaw(client.player.getYaw() - 5.0f); break;
            case AIM_RIGHT_SMALL: client.player.setYaw(client.player.getYaw() + 2.0f); break;
            case AIM_RIGHT_MEDIUM:client.player.setYaw(client.player.getYaw() + 5.0f); break;
            case AIM_UP_SMALL:    client.player.setPitch(client.player.getPitch() - 2.0f); break;
            case AIM_DOWN_SMALL:  client.player.setPitch(client.player.getPitch() + 2.0f); break;
            
            case ATTACK_SPAM: 
                performAttack(client);
                break;
                
            case ATTACK_TIMED_SWEEP:
                // 0.84 is the vanilla threshold for sweep attacks and near-max damage
                if (client.player.getAttackCooldownProgress(0.0f) >= 0.84f) {
                    performAttack(client);
                }
                break;
                
            case ATTACK_SPRINT:
                client.player.setSprinting(true);
                performAttack(client);
                break;
                
            case JUMP:
                if (client.player.isOnGround()) {
                    client.player.jump();
                }
                break;
                
            case BLOCK:
                client.options.useKey.setPressed(true); 
                break;
                
            case NO_OP: default: break;
        }
    }

    private void performAttack(MinecraftClient client) {
        if (client.crosshairTarget != null && client.crosshairTarget.getType() == HitResult.Type.ENTITY) {
            client.interactionManager.attackEntity(client.player, ((EntityHitResult)client.crosshairTarget).getEntity());
            client.player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
        }
    }
}
