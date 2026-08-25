package com.example.newgen6.rl.env;

import net.minecraft.block.BlockState;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class StateEncoder {

    public static final int SIZE = ActionSpace.STATE_SIZE;

    private static final int TERRAIN_RADIUS = 3; // -> 6x6 grid

    public float[] encode(ClientPlayerEntity self, LivingEntity target) {
        float[] s = new float[SIZE];
        int idx = 0;

        // ---- self status (16) ----
        s[idx++] = clamp01(self.getHealth() / self.getMaxHealth());
        s[idx++] = clamp01(self.getHungerManager().getFoodLevel() / 20f);
        s[idx++] = clamp01(self.getAir() / 300f);
        s[idx++] = wrapDegrees(self.getYaw()) / 180f;
        s[idx++] = clampPitch(self.getPitch()) / 90f;
        Vec3d selfVel = self.getVelocity();
        s[idx++] = (float) clampSym(selfVel.x, 1.0);
        s[idx++] = (float) clampSym(selfVel.y, 1.0);
        s[idx++] = (float) clampSym(selfVel.z, 1.0);
        s[idx++] = self.isOnGround() ? 1f : 0f;
        s[idx++] = self.isSprinting() ? 1f : 0f;
        s[idx++] = self.isSneaking() ? 1f : 0f;
        s[idx++] = clamp01(self.getAttackCooldownProgress(0f));
        s[idx++] = (float) (self.getY() / 320.0);
        s[idx++] = clamp01((float) (self.fallDistance / 20.0));
        s[idx++] = self.isOnFire() ? 1f : 0f;
        s[idx++] = 0f;

        // ---- target-relative info (14) ----
        boolean targetAlive = target != null && target.isAlive();
        if (targetAlive) {
            Vec3d rel = target.getPos().subtract(self.getPos());
            s[idx++] = (float) clampSym(rel.x, 32.0);
            s[idx++] = (float) clampSym(rel.y, 32.0);
            s[idx++] = (float) clampSym(rel.z, 32.0);
            s[idx++] = clamp01((float) (rel.length() / 32.0));
            double desiredYaw = Math.toDegrees(Math.atan2(-rel.x, rel.z));
            s[idx++] = angleDiff(self.getYaw(), (float) desiredYaw) / 180f;
            double horizDist = Math.sqrt(rel.x * rel.x + rel.z * rel.z);
            double desiredPitch = Math.toDegrees(-Math.atan2(rel.y, horizDist));
            s[idx++] = (float) clampSym(desiredPitch - self.getPitch(), 90.0);
            s[idx++] = clamp01(target.getHealth() / target.getMaxHealth());
            Vec3d tv = target.getVelocity();
            s[idx++] = (float) clampSym(tv.x, 1.0);
            s[idx++] = (float) clampSym(tv.y, 1.0);
            s[idx++] = (float) clampSym(tv.z, 1.0);
            s[idx++] = 1f;
            s[idx++] = target.isOnGround() ? 1f : 0f;
            s[idx++] = target.isSprinting() ? 1f : 0f;
            s[idx++] = 0f;
        } else {
            idx += 14;
        }

        // ---- combat context (8) ----
        double distance = targetAlive ? self.getPos().distanceTo(target.getPos()) : 999.0;
        s[idx++] = (targetAlive && canSee(self, target)) ? 1f : 0f;
        s[idx++] = (targetAlive && distance <= 3.0) ? 1f : 0f;
        
        s[idx++] = self.hurtTime > 0 ? (self.hurtTime / 10f) : 0f; 
        s[idx++] = (targetAlive && target.hurtTime > 0) ? (target.hurtTime / 10f) : 0f; 
        
        s[idx++] = clamp01(self.getAttackCooldownProgress(0f));
        s[idx++] = !self.getMainHandStack().isEmpty() ? 1f : 0f;
        s[idx++] = 0f;
        s[idx++] = self.isBlocking() ? 1f : 0f;

        // ---- local terrain grid, 6x6 (36) ----
        World world = self.getEntityWorld();
        BlockPos feet = self.getBlockPos();
        for (int dz = -TERRAIN_RADIUS; dz < TERRAIN_RADIUS; dz++) {
            for (int dx = -TERRAIN_RADIUS; dx < TERRAIN_RADIUS; dx++) {
                BlockPos p = feet.add(dx, 0, dz);
                s[idx++] = getRelativeHeight(world, p);
            }
        }

        // ---- misc / reserved (4) ----
        int aheadDx = (int) Math.round(-Math.sin(Math.toRadians(self.getYaw())));
        int aheadDz = (int) Math.round(Math.cos(Math.toRadians(self.getYaw())));
        BlockPos ahead = feet.add(aheadDx, 0, aheadDz);
        
        s[idx++] = getRelativeHeight(world, ahead) > 0f ? 1f : 0f;
        s[idx++] = isLedge(world, feet) ? 1f : 0f;
        s[idx++] = clamp01((self.age % 1200) / 1200f);
        s[idx++] = clamp01(world.getLightLevel(feet) / 15f);

        if (idx != SIZE) {
            throw new IllegalStateException("StateEncoder wrote " + idx + " floats, expected " + SIZE);
        }
        return s;
    }

    private static float getRelativeHeight(World world, BlockPos targetColumn) {
        for (int y = 3; y >= -3; y--) {
            BlockPos check = targetColumn.up(y);
            if (world.getBlockState(check).isSolidBlock(world, check)) {
                return (float) y / 3.0f; 
            }
        }
        return -1.0f; 
    }

    private static boolean isLedge(World world, BlockPos feet) {
        return world.getBlockState(feet.down(2)).isAir();
    }

    private static boolean canSee(ClientPlayerEntity self, LivingEntity target) {
        return self.canSee(target);
    }

    private static float wrapDegrees(float deg) {
        float d = deg % 360f;
        if (d >= 180f) d -= 360f;
        if (d < -180f) d += 360f;
        return d;
    }

    private static float clampPitch(float p) {
        return Math.max(-90f, Math.min(90f, p));
    }

    private static double clampSym(double v, double bound) {
        return Math.max(-bound, Math.min(bound, v)) / bound;
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
    }

    private static float angleDiff(float a, float b) {
        return wrapDegrees(b - a);
    }
}
