package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.SwordItem;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.Arrays;

public final class ObservationBuilder {
    public final float[] observation = new float[AgentConfig.OBS_DIM];

    private final float[] history = new float[AgentConfig.HISTORY_TICKS * AgentConfig.OBS_DIM];
    private int historyWrite = 0;
    private int historyCount = 0;

    private LivingEntity target;
    private int episodeTick = 0;

    private double prevPlayerTotalHealth = 0.0;
    private double prevTargetTotalHealth = 0.0;

    private float tickDamageDealt = 0.0f;
    private float tickDamageTaken = 0.0f;

    private boolean tickWin = false;
    private boolean tickHit = false;
    private boolean tickMiss = false;

    private float damageDealtEma = 0.0f;
    private float damageTakenEma = 0.0f;
    private float attacksEma = 0.0f;
    private float hitsEma = 0.0f;
    private float missEma = 0.0f;

    private int lastAttackTick = -1000;
    private int lastHitTick = -1000;
    private int lastDamageTakenTick = -1000;
    private int lastMissCheckAttack = -1000;

    private float lastDistance = 0.0f;

    // Player dynamics memory
    private float prevPlayerSpeed = 0.0f;
    private float playerAccelEma = 0.0f;
    private float prevYaw = 0.0f;
    private float prevPitch = 0.0f;
    private float yawRateEma = 0.0f;
    private float pitchRateEma = 0.0f;
    private float prevPlayerVelX = 0.0f;
    private float prevPlayerVelZ = 0.0f;
    private float playerKnockX = 0.0f;
    private float playerKnockZ = 0.0f;

    // Health trend memory
    private float prevPlayerHealthNorm = -1.0f;
    private float playerHealthEma = 0.0f;
    private float playerHealthDelta = 0.0f;

    private float prevTargetHealthNorm = -1.0f;
    private float targetHealthEma = 0.0f;
    private float targetHealthDelta = 0.0f;

    // Target dynamics memory
    private float prevTargetSpeed = 0.0f;
    private float targetAccelEma = 0.0f;
    private float targetSpeedEma = 0.0f;
    private float targetVelYema = 0.0f;

    private float prevDist = 0.0f;
    private float closingSpeedEma = 0.0f;
    private float distEma = 0.0f;

    private float prevRelYaw = 0.0f;
    private float relYawEma = 0.0f;
    private float relYawRateEma = 0.0f;

    private float prevTargetVelX = 0.0f;
    private float prevTargetVelZ = 0.0f;
    private float targetKnockX = 0.0f;
    private float targetKnockZ = 0.0f;

    private int prevTargetHurtTime = 0;

    private float targetOnGroundEma = 0.0f;
    private float targetSprintEma = 0.0f;

    private float prevTargetLateral = 0.0f;
    private float targetStrafeChangeEma = 0.0f;

    private float prevTargetCooldown = 0.0f;
    private float targetCooldownRate = 0.0f;

    private float targetHeadingPrev = 0.0f;
    private float targetHeadingRateEma = 0.0f;

    private int timeSinceTarget = 0;

    public void update(MinecraftClient client) {
        if (client.world == null || client.player == null) return;

        tickDamageDealt = 0.0f;
        tickDamageTaken = 0.0f;
        tickWin = false;
        tickHit = false;
        tickMiss = false;

        episodeTick++;

        updateTarget(client);
        updateDamageTracking(client);

        Arrays.fill(observation, 0.0f);

        fillPlayer(client);
        fillTarget(client);
        fillCombat(client);
        fillEnvironment(client);
        fillTemporal();

        writeHistory();
    }

    private void updateTarget(MinecraftClient client) {
        ClientPlayerEntity player = client.player;

        if (target != null && target.getHealth() <= 0.0f) {
            tickWin = true;
        }

        LivingEntity best = null;
        double bestDist = 48.0;

        for (Entity e : client.world.getEntities()) {
            if (!(e instanceof LivingEntity living)) continue;
            if (living == player) continue;
            if (living.getHealth() <= 0.0f) continue;

            if (living instanceof PlayerEntity pe && pe.isSpectator()) {
                continue;
            }

            double d = player.distanceTo(living);
            if (d > 48.0) continue;

            if (d < bestDist) {
                bestDist = d;
                best = living;
            }
        }

        if (best != target) {
            target = best;

            if (target != null) {
                prevTargetTotalHealth = target.getHealth() + target.getAbsorptionAmount();

                prevTargetHealthNorm = clamp(target.getHealth() / Math.max(1.0f, target.getMaxHealth()), 0.0f, 1.0f);
                targetHealthEma = prevTargetHealthNorm;
                targetHealthDelta = 0.0f;

                prevTargetSpeed = 0.0f;
                targetAccelEma = 0.0f;
                targetSpeedEma = 0.0f;
                targetVelYema = 0.0f;

                prevDist = 0.0f;
                closingSpeedEma = 0.0f;
                distEma = 0.0f;

                prevRelYaw = 0.0f;
                relYawEma = 0.0f;
                relYawRateEma = 0.0f;

                Vec3d tv = target.getVelocity();
                prevTargetVelX = (float) tv.x;
                prevTargetVelZ = (float) tv.z;
                targetKnockX = 0.0f;
                targetKnockZ = 0.0f;

                prevTargetHurtTime = target.hurtTime;

                targetOnGroundEma = target.isOnGround() ? 1.0f : 0.0f;
                targetSprintEma = target.isSprinting() ? 1.0f : 0.0f;

                prevTargetLateral = 0.0f;
                targetStrafeChangeEma = 0.0f;

                prevTargetCooldown = 0.0f;
                targetCooldownRate = 0.0f;

                targetHeadingPrev = 0.0f;
                targetHeadingRateEma = 0.0f;

                timeSinceTarget = 0;
            } else {
                prevTargetTotalHealth = 0.0;
                prevTargetHealthNorm = -1.0f;
            }
        }

        if (target == null) {
            timeSinceTarget++;
            decayTargetMemory();
        } else {
            timeSinceTarget = 0;
        }
    }

    private void decayTargetMemory() {
        targetHealthEma *= 0.95f;
        targetHealthDelta *= 0.95f;

        targetAccelEma *= 0.95f;
        targetSpeedEma *= 0.95f;
        targetVelYema *= 0.95f;

        closingSpeedEma *= 0.95f;
        distEma *= 0.95f;

        relYawEma *= 0.95f;
        relYawRateEma *= 0.95f;

        targetKnockX *= 0.95f;
        targetKnockZ *= 0.95f;

        targetOnGroundEma *= 0.95f;
        targetSprintEma *= 0.95f;

        targetStrafeChangeEma *= 0.95f;
        targetCooldownRate *= 0.95f;
        targetHeadingRateEma *= 0.95f;

        lastDistance = 0.0f;
    }

    private void updateDamageTracking(MinecraftClient client) {
        ClientPlayerEntity player = client.player;

        float playerTotal = player.getHealth() + player.getAbsorptionAmount();

        if (prevPlayerTotalHealth > 0.0 && playerTotal < prevPlayerTotalHealth - 0.01) {
            float dmg = (float) (prevPlayerTotalHealth - playerTotal);
            tickDamageTaken += dmg;
            lastDamageTakenTick = episodeTick;
            damageTakenEma = damageTakenEma * 0.9f + Math.min(1.0f, dmg * 0.1f);
        }

        prevPlayerTotalHealth = playerTotal;

        float maxPlayerHealth = Math.max(1.0f, player.getMaxHealth());
        float playerHealthNorm = clamp(player.getHealth() / maxPlayerHealth, 0.0f, 1.0f);

        if (prevPlayerHealthNorm < -0.5f) {
            prevPlayerHealthNorm = playerHealthNorm;
            playerHealthEma = playerHealthNorm;
        } else {
            float delta = playerHealthNorm - prevPlayerHealthNorm;
            playerHealthDelta = playerHealthDelta * 0.8f + clamp(delta, -0.5f, 0.5f) * 0.2f;
            prevPlayerHealthNorm = playerHealthNorm;
            playerHealthEma = playerHealthEma * 0.9f + playerHealthNorm * 0.1f;
        }

        if (target != null) {
            float targetTotal = target.getHealth() + target.getAbsorptionAmount();

            if (prevTargetTotalHealth > 0.0 && targetTotal < prevTargetTotalHealth - 0.01) {
                float dmg = (float) (prevTargetTotalHealth - targetTotal);

                if (episodeTick - lastAttackTick <= 12) {
                    tickDamageDealt += dmg;
                    damageDealtEma = damageDealtEma * 0.9f + Math.min(1.0f, dmg * 0.1f);

                    if (lastHitTick < lastAttackTick) {
                        lastHitTick = episodeTick;
                        hitsEma = hitsEma * 0.9f + 0.1f;
                        tickHit = true;
                    }
                }
            }

            prevTargetTotalHealth = targetTotal;

            if (lastAttackTick > lastMissCheckAttack &&
                episodeTick - lastAttackTick >= 12 &&
                lastHitTick < lastAttackTick) {
                tickMiss = true;
                lastMissCheckAttack = lastAttackTick;
            }
        } else {
            prevTargetTotalHealth = 0.0;
        }

        if (tickMiss) {
            missEma = missEma * 0.9f + 0.1f;
        } else {
            missEma *= 0.98f;
        }

        damageDealtEma *= 0.98f;
        damageTakenEma *= 0.98f;
        attacksEma *= 0.98f;
        hitsEma *= 0.98f;
    }

    private void fillPlayer(MinecraftClient client) {
        ClientPlayerEntity p = client.player;

        observation[0] = clamp(p.getHealth() / Math.max(1.0f, p.getMaxHealth()), 0.0f, 1.0f);
        observation[1] = clamp(p.getAbsorptionAmount() / 20.0f, 0.0f, 2.0f);
        observation[2] = clamp(p.getHungerManager().getFoodLevel() / 20.0f, 0.0f, 1.0f);
        observation[3] = clamp(p.getHungerManager().getSaturationLevel() / 20.0f, 0.0f, 1.0f);
        observation[4] = clamp(p.getAir() / 300.0f, 0.0f, 1.0f);
        observation[5] = p.isOnGround() ? 1.0f : 0.0f;
        observation[6] = p.isSprinting() ? 1.0f : 0.0f;
        observation[7] = p.isSneaking() ? 1.0f : 0.0f;
        observation[8] = clamp(p.fallDistance / 16.0f, -2.0f, 2.0f);

        Vec3d v = p.getVelocity();
        observation[9] = clamp((float) v.x / 10.0f, -2.0f, 2.0f);
        observation[10] = clamp((float) v.y / 10.0f, -2.0f, 2.0f);
        observation[11] = clamp((float) v.z / 10.0f, -2.0f, 2.0f);

        float speed = (float) Math.sqrt(v.x * v.x + v.z * v.z);
        float speedNorm = clamp(speed / 10.0f, 0.0f, 2.0f);
        observation[12] = speedNorm;

        observation[13] = wrapDeg(p.getYaw()) / 180.0f;
        observation[14] = clamp(p.getPitch() / 90.0f, -1.0f, 1.0f);

        float cd = p.getAttackCooldownProgress(0.0f);
        observation[15] = clamp(cd, 0.0f, 1.0f);
        observation[16] = cd >= 1.0f ? 1.0f : 0.0f;

        observation[17] = clamp(p.hurtTime / 10.0f, 0.0f, 1.0f);
        observation[18] = 0.0f;
        observation[19] = clamp(damageTakenEma, 0.0f, 1.0f);

        observation[20] = p.input.pressingForward ? 1.0f : 0.0f;
        observation[21] = p.input.pressingBack ? 1.0f : 0.0f;
        observation[22] = p.input.pressingLeft ? 1.0f : 0.0f;
        observation[23] = p.input.pressingRight ? 1.0f : 0.0f;

        observation[24] = client.options.jumpKey.isPressed() ? 1.0f : 0.0f;
        observation[25] = client.options.sneakKey.isPressed() ? 1.0f : 0.0f;
        observation[26] = client.options.sprintKey.isPressed() ? 1.0f : 0.0f;

        observation[27] = p.isTouchingWater() ? 1.0f : 0.0f;
        observation[28] = p.isOnFire() ? 1.0f : 0.0f;
        observation[29] = p.getMainHandStack().getItem() instanceof SwordItem ? 1.0f : 0.0f;

        observation[30] = clamp(episodeTick / 6000.0f, 0.0f, 1.0f);

        float yawRad = (float) Math.toRadians(p.getYaw());
        float pitchRad = (float) Math.toRadians(p.getPitch());

        observation[31] = (float) Math.sin(yawRad);
        observation[32] = (float) Math.cos(yawRad);
        observation[33] = (float) Math.sin(pitchRad);
        observation[34] = (float) Math.cos(pitchRad);

        // Enhanced player dynamics
        float accel = speedNorm - prevPlayerSpeed;
        playerAccelEma = playerAccelEma * 0.8f + clamp(accel, -1.0f, 1.0f) * 0.2f;
        prevPlayerSpeed = speedNorm;
        observation[35] = clamp(playerAccelEma, -1.0f, 1.0f);

        float yawDelta = wrapDeg(p.getYaw() - prevYaw);
        yawRateEma = yawRateEma * 0.8f + clamp(yawDelta / 25.0f, -1.0f, 1.0f) * 0.2f;
        prevYaw = p.getYaw();
        observation[36] = clamp(yawRateEma, -1.0f, 1.0f);

        float pitchDelta = wrapDeg(p.getPitch() - prevPitch);
        pitchRateEma = pitchRateEma * 0.8f + clamp(pitchDelta / 25.0f, -1.0f, 1.0f) * 0.2f;
        prevPitch = p.getPitch();
        observation[37] = clamp(pitchRateEma, -1.0f, 1.0f);

        float fx = -yawRadSin(yawRad);
        float fz = yawRadCos(yawRad);

        float forwardSpeed = (float) (v.x * fx + v.z * fz) / 10.0f;
        float lateralSpeed = (float) (v.x * fz - v.z * fx) / 10.0f;

        observation[38] = clamp(forwardSpeed, -2.0f, 2.0f);
        observation[39] = clamp(lateralSpeed, -2.0f, 2.0f);

        // Player knockback estimate
        float pkx = (float) v.x - prevPlayerVelX;
        float pkz = (float) v.z - prevPlayerVelZ;
        playerKnockX = playerKnockX * 0.8f + clamp(pkx, -1.0f, 1.0f) * 0.2f;
        playerKnockZ = playerKnockZ * 0.8f + clamp(pkz, -1.0f, 1.0f) * 0.2f;
        prevPlayerVelX = (float) v.x;
        prevPlayerVelZ = (float) v.z;
    }

    private void fillTarget(MinecraftClient client) {
        if (target == null) {
            lastDistance = 0.0f;
            return;
        }

        ClientPlayerEntity p = client.player;

        Vec3d rel = target.getPos().subtract(p.getPos());

        observation[40] = 1.0f;
        observation[41] = clamp((float) rel.x / 16.0f, -2.0f, 2.0f);
        observation[42] = clamp((float) rel.y / 16.0f, -2.0f, 2.0f);
        observation[43] = clamp((float) rel.z / 16.0f, -2.0f, 2.0f);

        double horizontal = Math.hypot(rel.x, rel.z);
        observation[44] = clamp((float) horizontal / 16.0f, 0.0f, 2.0f);
        observation[45] = clamp((float) rel.y / 8.0f, -2.0f, 2.0f);

        double dist = Math.sqrt(rel.lengthSquared());
        lastDistance = (float) dist;
        observation[46] = clamp((float) dist / 16.0f, 0.0f, 2.0f);

        Vec3d tv = target.getVelocity();
        observation[47] = clamp((float) tv.x / 10.0f, -2.0f, 2.0f);
        observation[48] = clamp((float) tv.y / 10.0f, -2.0f, 2.0f);
        observation[49] = clamp((float) tv.z / 10.0f, -2.0f, 2.0f);

        float targetSpeed = (float) Math.sqrt(tv.x * tv.x + tv.z * tv.z);
        float targetSpeedNorm = clamp(targetSpeed / 10.0f, 0.0f, 2.0f);
        observation[50] = targetSpeedNorm;

        float maxTargetHealth = Math.max(1.0f, target.getMaxHealth());
        float targetHealthNorm = clamp(target.getHealth() / maxTargetHealth, 0.0f, 1.0f);
        observation[51] = targetHealthNorm;

        observation[52] = clamp(target.getAbsorptionAmount() / 20.0f, 0.0f, 2.0f);
        observation[53] = target.isOnGround() ? 1.0f : 0.0f;
        observation[54] = target.isSprinting() ? 1.0f : 0.0f;
        observation[55] = target.isSneaking() ? 1.0f : 0.0f;
        observation[56] = clamp(target.hurtTime / 10.0f, 0.0f, 1.0f);

        observation[57] = wrapDeg(target.getYaw()) / 180.0f;
        observation[58] = clamp(target.getPitch() / 90.0f, -1.0f, 1.0f);

        float desiredYaw = (float) Math.toDegrees(Math.atan2(rel.x, rel.z));
        float relYaw = wrapDeg(desiredYaw - p.getYaw());
        observation[59] = relYaw / 180.0f;

        float desiredPitch = (float) Math.toDegrees(Math.atan2(-rel.y, Math.max(0.001, horizontal)));
        float relPitch = wrapDeg(desiredPitch - p.getPitch());
        observation[60] = clamp(relPitch / 90.0f, -1.0f, 1.0f);

        observation[61] = dist <= 3.2 ? 1.0f : 0.0f;
        observation[62] = p.canSee(target) ? 1.0f : 0.0f;
        observation[63] = (float) Math.cos(Math.toRadians(relYaw)) > 0.0f ? 1.0f : 0.0f;

        double dot = rel.x * tv.x + rel.z * tv.z;
        observation[64] = dot > 0.05 ? 1.0f : 0.0f;   // moving away
        observation[65] = dot < -0.05 ? 1.0f : 0.0f;  // moving toward

        // Player-relative lateral target motion
        float yawRad = (float) Math.toRadians(p.getYaw());
        float fx = -yawRadSin(yawRad);
        float fz = yawRadCos(yawRad);

        float targetLateral = (float) (tv.x * fz - tv.z * fx) / 10.0f;
        float targetForward = (float) (tv.x * fx + tv.z * fz) / 10.0f;

        observation[66] = targetLateral < -0.02f ? 1.0f : 0.0f;
        observation[67] = targetLateral > 0.02f ? 1.0f : 0.0f;

        observation[68] = tv.y > 0.08 ? 1.0f : 0.0f;
        observation[69] = tv.y < -0.08 ? 1.0f : 0.0f;

        // Target acceleration / closing speed / distance memory
        float targetAccel = targetSpeedNorm - prevTargetSpeed;
        targetAccelEma = targetAccelEma * 0.8f + clamp(targetAccel, -1.0f, 1.0f) * 0.2f;
        prevTargetSpeed = targetSpeedNorm;
        observation[70] = clamp(targetAccelEma, -1.0f, 1.0f);

        float distNorm = clamp((float) dist / 16.0f, 0.0f, 2.0f);
        float closing = (float) dist - prevDist;
        closingSpeedEma = closingSpeedEma * 0.8f + clamp(closing, -1.0f, 1.0f) * 0.2f;
        prevDist = (float) dist;
        observation[71] = clamp(closingSpeedEma * 2.0f, -1.0f, 1.0f);

        distEma = distEma * 0.9f + distNorm * 0.1f;
        observation[72] = clamp(distEma, 0.0f, 2.0f);

        if (target instanceof PlayerEntity pe) {
            float targetCd = clamp(pe.getAttackCooldownProgress(0.0f), 0.0f, 1.0f);
            observation[73] = targetCd;

            targetCooldownRate = targetCooldownRate * 0.8f + clamp(targetCd - prevTargetCooldown, -1.0f, 1.0f) * 0.2f;
            prevTargetCooldown = targetCd;
        } else {
            prevTargetCooldown = 0.0f;
            targetCooldownRate *= 0.9f;
        }

        observation[74] = target.getMainHandStack().getItem() instanceof SwordItem ? 1.0f : 0.0f;

        // Relative yaw memory
        relYawEma = relYawEma * 0.9f + (relYaw / 180.0f) * 0.1f;
        observation[75] = clamp(relYawEma, -1.0f, 1.0f);

        float relYawRate = wrapDeg(relYaw - prevRelYaw);
        relYawRateEma = relYawRateEma * 0.8f + clamp(relYawRate / 25.0f, -1.0f, 1.0f) * 0.2f;
        prevRelYaw = relYaw;
        observation[76] = clamp(relYawRateEma, -1.0f, 1.0f);

        observation[77] = clamp(targetLateral, -2.0f, 2.0f);
        observation[78] = clamp(targetForward, -2.0f, 2.0f);

        float relYawRad = (float) Math.toRadians(relYaw);
        observation[79] = clamp((float) Math.sin(relYawRad), -1.0f, 1.0f);
        observation[80] = clamp((float) Math.cos(relYawRad) - 1.0f, -2.0f, 0.0f);

        // Target heading change
        if (targetSpeed > 0.05f) {
            float heading = (float) Math.toDegrees(Math.atan2(tv.x, tv.z));
            float headingDelta = wrapDeg(heading - targetHeadingPrev);
            targetHeadingRateEma = targetHeadingRateEma * 0.8f + clamp(headingDelta / 25.0f, -1.0f, 1.0f) * 0.2f;
            targetHeadingPrev = heading;
        } else {
            targetHeadingRateEma *= 0.9f;
        }
        observation[81] = clamp(targetHeadingRateEma, -1.0f, 1.0f);

        // Target knockback estimate
        float tkx = (float) tv.x - prevTargetVelX;
        float tkz = (float) tv.z - prevTargetVelZ;
        targetKnockX = targetKnockX * 0.8f + clamp(tkx, -1.0f, 1.0f) * 0.2f;
        targetKnockZ = targetKnockZ * 0.8f + clamp(tkz, -1.0f, 1.0f) * 0.2f;
        prevTargetVelX = (float) tv.x;
        prevTargetVelZ = (float) tv.z;

        observation[82] = clamp(targetKnockX, -1.0f, 1.0f);
        observation[83] = clamp(ta