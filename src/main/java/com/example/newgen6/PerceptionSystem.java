package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;

public class PerceptionSystem {

    /**
     * Extracts a 16-element normalized observation array from the current game state.
     * All values are heavily normalized to stay roughly between -1.0 and 1.0, 
     * which allows the DDPG neural network to converge much faster.
     */
    public static float[] getObservation(MinecraftClient client, LivingEntity target) {
        float[] state = new float[16];

        if (client.player == null) {
            return state; // Return array of 0s if the player doesn't exist yet
        }

        // --- SECTION 1: Player Self-State (Always available) ---
        state[3] = client.player.getHealth() / 20.0f;                               // 3: My Health (0.0 to 1.0)
        state[5] = client.player.getAttackCooldownProgress(0.0f);                   // 5: Weapon Cooldown (0.0 to 1.0)
        
        Vec3d myVel = client.player.getVelocity();
        state[6] = (float) MathHelper.clamp(myVel.x / 2.0, -1.0, 1.0);              // 6: My X Velocity
        state[7] = (float) MathHelper.clamp(myVel.y / 2.0, -1.0, 1.0);              // 7: My Y Velocity (Falling/Jumping)
        state[8] = (float) MathHelper.clamp(myVel.z / 2.0, -1.0, 1.0);              // 8: My Z Velocity
        
        state[12] = client.player.isOnGround() ? 1.0f : 0.0f;                       // 12: Am I on the ground?
        state[14] = client.player.hurtTime / 10.0f;                                 // 14: My Hurt Time (Damage blink indicator)

        // --- SECTION 2: Target-Relative State (Requires enemy) ---
        if (target != null && target.isAlive()) {
            // Distance and Orientation
            float distance = (float) client.player.distanceTo(target);
            state[0] = MathHelper.clamp(distance / 20.0f, 0.0f, 1.0f);              // 0: Normalized Distance (Capped at 20 blocks)

            float targetYaw = calculateYawTo(client.player, target);
            float targetPitch = calculatePitchTo(client.player, target);
            
            float yawError = MathHelper.wrapDegrees(targetYaw - client.player.getYaw());
            float pitchError = MathHelper.wrapDegrees(targetPitch - client.player.getPitch());

            state[1] = yawError / 180.0f;                                           // 1: Yaw Error (-1.0 to 1.0)
            state[2] = pitchError / 90.0f;                                          // 2: Pitch Error (-1.0 to 1.0)
            
            // Target specific stats
            state[4] = target.getHealth() / 20.0f;                                  // 4: Enemy Health
            
            Vec3d targetVel = target.getVelocity();
            state[9] = (float) MathHelper.clamp(targetVel.x / 2.0, -1.0, 1.0);      // 9: Enemy X Velocity
            state[10] = (float) MathHelper.clamp(targetVel.y / 2.0, -1.0, 1.0);     // 10: Enemy Y Velocity
            state[11] = (float) MathHelper.clamp(targetVel.z / 2.0, -1.0, 1.0);     // 11: Enemy Z Velocity
            
            state[13] = target.isOnGround() ? 1.0f : 0.0f;                          // 13: Is Enemy on the ground?
            state[15] = target.hurtTime / 10.0f;                                    // 15: Enemy Hurt Time
        } else {
            // Fallback defaults if no target is found (Prevents array index bounds or NaN errors)
            state[0] = 1.0f;  // Max distance
            state[1] = 0.0f;
            state[2] = 0.0f;
            state[4] = 0.0f;  // Enemy health 0
            state[9] = 0.0f;
            state[10] = 0.0f;
            state[11] = 0.0f;
            state[13] = 0.0f;
            state[15] = 0.0f;
        }

        return state;
    }

    /**
     * Calculates the exact horizontal Yaw angle required to look directly at the target's center.
     */
    private static float calculateYawTo(LivingEntity player, LivingEntity target) {
        double dx = target.getX() - player.getX();
        double dz = target.getZ() - player.getZ();
        return (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0);
    }

    /**
     * Calculates the exact vertical Pitch angle required to look at the target's eye level.
     */
    private static float calculatePitchTo(LivingEntity player, LivingEntity target) {
        double dx = target.getX() - player.getX();
        double dz = target.getZ() - player.getZ();
        double distance = Math.sqrt(dx * dx + dz * dz);
        
        // Aim for the eyes / upper body
        double targetHeight = target.getEyeY() - (target.getHeight() * 0.2); 
        double dy = targetHeight - player.getEyeY();
        
        return (float) -Math.toDegrees(Math.atan2(dy, distance));
    }
}
