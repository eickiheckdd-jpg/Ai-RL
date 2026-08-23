package com.example.newgen6;

import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;

public class CombatObservationBuilder {

    public static float[] buildObservation(PlayerEntity agent, LivingEntity target) {
        float[] obs = new float[RLConfig.OBS_DIM];
        if (agent == null || target == null || !target.isAlive()) return obs;

        HitResult hit = agent.raycast(20.0D, 1.0F, false);
        boolean targetPresent = (hit.getType() == HitResult.Type.ENTITY) && (((EntityHitResult) hit).getEntity() == target);
        obs[0] = targetPresent ? 1.0f : 0.0f;

        Vec3d toTarget = target.getEyePos().subtract(agent.getEyePos()).normalize();
        double yawDiff = MathHelper.wrapDegrees(agent.getYaw() - (float) Math.toDegrees(Math.atan2(-toTarget.x, toTarget.z)));
        double pitchDiff = MathHelper.wrapDegrees(agent.getPitch() - (float) Math.toDegrees(-Math.atan2(toTarget.y, Math.hypot(toTarget.x, toTarget.z))));
        
        obs[1] = sanitize((float) (yawDiff / 90.0));
        obs[2] = sanitize((float) (pitchDiff / 90.0));

        float dist = (float) agent.distanceTo(target);
        obs[3] = sanitize(dist / 20.0f);
        obs[4] = sanitize(target.getWidth() / 2.0f);
        obs[5] = sanitize(target.getHeight() / 2.0f);
        obs[6] = sanitize((float) target.getEyeY() - (float) agent.getEyeY());

        Vec3d relVel = target.getVelocity().subtract(agent.getVelocity());
        obs[7] = sanitize((float) relVel.x);
        obs[8] = sanitize((float) relVel.y);
        obs[9] = sanitize((float) relVel.z);

        obs[10] = sanitize(agent.getHealth() / agent.getMaxHealth());
        obs[11] = sanitize(agent.getHungerManager().getFoodLevel() / 20.0f);
        obs[12] = sanitize(target.getHealth() / target.getMaxHealth());
        obs[13] = sanitize((agent.getPitch() % 360.0f) / 180.0f);

        return obs;
    }

    private static float sanitize(float val) {
        return (Float.isNaN(val) || Float.isInfinite(val)) ? 0.0f : MathHelper.clamp(val, -1.0f, 1.0f);
    }
}
