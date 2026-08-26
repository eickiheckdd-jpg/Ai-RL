package com.example.newgen6.rl;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

import java.util.Arrays;

public class Observation {
    public static final int SIZE = 229;

    // Persistent static array to ensure ZERO heap allocations per tick
    private final float[] buffer = new float[SIZE];

    public float[] extract(ClientPlayerEntity player, Entity target) {
        // Clear previous state safely
        Arrays.fill(buffer, 0.0f);

        if (player == null || target == null) {
            return buffer; 
        }

        World world = player.getWorld();

        // ---------------------------------------------------------
        // SECTION 1: KINEMATICS & AIM GEOMETRY (Indices 0 - 11)
        // ---------------------------------------------------------
        double dx = target.getX() - player.getX();
        double dy = target.getEyeY() - player.getEyeY();
        double dz = target.getZ() - player.getZ();
        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Required absolute angles to look at the target center
        float requiredYaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float requiredPitch = (float) Math.toDegrees(-Math.atan2(dy, Math.sqrt(dx * dx + dz * dz)));

        // Angular error mapped to [-180, 180]
        float yawError = MathHelper.wrapDegrees(requiredYaw - player.getYaw());
        float pitchError = MathHelper.wrapDegrees(requiredPitch - player.getPitch());

        Vec3d selfVel = player.getVelocity();
        Vec3d targetVel = target.getVelocity();

        // Normalized to bounded floats [-1.0f, 1.0f]
        buffer[0] = (float) MathHelper.clamp(dx / 32.0, -1.0, 1.0);
        buffer[1] = (float) MathHelper.clamp(dy / 32.0, -1.0, 1.0);
        buffer[2] = (float) MathHelper.clamp(dz / 32.0, -1.0, 1.0);
        buffer[3] = (float) MathHelper.clamp(distance / 32.0, 0.0, 1.0);
        buffer[4] = MathHelper.clamp(yawError / 180.0f, -1.0f, 1.0f);
        buffer[5] = MathHelper.clamp(pitchError / 90.0f, -1.0f, 1.0f);

        buffer[6] = (float) MathHelper.clamp(selfVel.x / 2.0, -1.0, 1.0);
        buffer[7] = (float) MathHelper.clamp(selfVel.y / 2.0, -1.0, 1.0);
        buffer[8] = (float) MathHelper.clamp(selfVel.z / 2.0, -1.0, 1.0);

        buffer[9] = (float) MathHelper.clamp(targetVel.x / 2.0, -1.0, 1.0);
        buffer[10] = (float) MathHelper.clamp(targetVel.y / 2.0, -1.0, 1.0);
        buffer[11] = (float) MathHelper.clamp(targetVel.z / 2.0, -1.0, 1.0);

        // ---------------------------------------------------------
        // SECTION 2: COMBAT STATUS (Indices 12 - 31)
        // ---------------------------------------------------------
        buffer[12] = player.getHealth() / player.getMaxHealth();
        buffer[13] = player.getHungerManager().getFoodLevel() / 20.0f;
        buffer[14] = player.isOnGround() ? 1.0f : 0.0f;
        buffer[15] = player.isSprinting() ? 1.0f : 0.0f;
        buffer[16] = player.isSneaking() ? 1.0f : 0.0f;
        buffer[17] = MathHelper.clamp(player.getAttackCooldownProgress(0.0f), 0.0f, 1.0f);
        buffer[18] = MathHelper.clamp(player.hurtTime / 10.0f, 0.0f, 1.0f);

        if (target instanceof LivingEntity livingTarget) {
            buffer[19] = livingTarget.getHealth() / livingTarget.getMaxHealth();
            buffer[20] = livingTarget.isOnGround() ? 1.0f : 0.0f;
            buffer[21] = livingTarget.isSprinting() ? 1.0f : 0.0f;
            buffer[22] = MathHelper.clamp(livingTarget.hurtTime / 10.0f, 0.0f, 1.0f);
            buffer[23] = livingTarget.isBlocking() ? 1.0f : 0.0f;
        }

        // ---------------------------------------------------------
        // SECTION 3: SPATIAL RAYCAST GRID (Indices 32 - 159)
        // ---------------------------------------------------------
        int rayIdx = 32;
        for (int i = 0; i < 128; i++) {
            // Fibonacci sphere distribution for optimal coverage
            double phi = Math.acos(1.0 - 2.0 * (i + 0.5) / 128.0);
            double theta = Math.PI * (1.0 + Math.sqrt(5.0)) * (i + 0.5);

            double rx = Math.sin(phi) * Math.cos(theta);
            double ry = Math.cos(phi);
            double rz = Math.sin(phi) * Math.sin(theta);

            double hitDist = 8.0; 
            for (double d = 0.5; d <= 8.0; d += 0.5) {
                if (!world.getBlockState(player.getBlockPos().add(
                        (int)(rx * d), (int)(ry * d), (int)(rz * d))).isAir()) {
                    hitDist = d;
                    break;
                }
            }
            buffer[rayIdx++] = (float) (hitDist / 8.0);
        }

        // Indices 160 - 228 remain initialized to 0.0f (Reserved for Action Memory)
        return buffer;
    }

    public float getYawErrorDegrees() {
        return buffer[4] * 180.0f;
    }

    public float getPitchErrorDegrees() {
        return buffer[5] * 90.0f;
    }
}
