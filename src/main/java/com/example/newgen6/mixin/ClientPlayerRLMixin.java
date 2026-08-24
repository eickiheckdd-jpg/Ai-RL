package com.example.newgen6.mixin;

import com.example.newgen6.NewGen6RLMod;
import com.example.newgen6.rl.PPOAgent.StepData;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.StreamSupport;

@Mixin(ClientPlayerEntity.class)
public abstract class ClientPlayerRLMixin {

    @Unique private final List<StepData> rlMemory = new ArrayList<>();
    @Unique private UUID lastTargetUuid = null;
    @Unique private float prevTargetHealth = 20f;
    @Unique private float prevSelfHealth = 20f;

    @Inject(method = "tick", at = @At("HEAD"))
    private void onClientTick(CallbackInfo ci) {
        if (!NewGen6RLMod.aiEnabled) return;

        ClientPlayerEntity player = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.world == null || player.isDead()) return;

        LivingEntity target = getNearestTarget(player, client);
        if (target == null) return; 

        float[] state = extract30DState(player, target);

        float[] continuousActions = new float[2];
        int[] discreteActions = new int[7]; // [W, S, A, D, Jump, Sprint, Attack]
        float[] logProb = new float[1];
        float[] value = new float[1];

        NewGen6RLMod.AGENT.selectAction(state, continuousActions, discreteActions, logProb, value);

        applyInputs(player, client, continuousActions, discreteActions);

        float reward = calculateTacticalReward(player, target);
        boolean done = target.isDead() || player.isDead();

        rlMemory.add(new StepData(state, continuousActions[0], continuousActions[1], discreteActions, logProb[0], reward, value[0], done));

        if (rlMemory.size() >= 128 || done) {
            NewGen6RLMod.AGENT.train(rlMemory);
            rlMemory.clear();
        }
    }

    @Unique
    private float[] extract30DState(ClientPlayerEntity p, LivingEntity t) {
        Vec3d pV = p.getVelocity();
        Vec3d tV = t.getVelocity();

        double dx = t.getX() - p.getX();
        double dy = t.getEyeY() - p.getEyeY();
        double dz = t.getZ() - p.getZ();
        double distXZ = Math.sqrt(dx * dx + dz * dz);
        double dist = p.distanceTo(t);

        float targetYaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0f);
        float targetPitch = (float) (-Math.toDegrees(Math.atan2(dy, distXZ)));

        float yawDiff = MathHelper.wrapDegrees(targetYaw - p.getYaw());
        float pitchDiff = MathHelper.wrapDegrees(targetPitch - p.getPitch());

        // Facing vectors and alignment
        Vec3d targetLook = t.getRotationVector();
        Vec3d toPlayer = new Vec3d(-dx, -dy, -dz).normalize();
        double targetFacingDot = targetLook.dotProduct(toPlayer);

        Vec3d playerLook = p.getRotationVector();
        Vec3d toTarget = new Vec3d(dx, dy, dz).normalize();
        double playerLookDot = playerLook.dotProduct(toTarget);

        Vec3d relVel = pV.subtract(tV);
        double closingSpeed = relVel.dotProduct(toTarget);

        return new float[] {
            (float) pV.x, (float) pV.y, (float) pV.z,
            (float) dx, (float) dy, (float) dz,
            (float) tV.x, (float) tV.y, (float) tV.z,
            yawDiff / 180f, pitchDiff / 90f,
            (float) targetFacingDot,
            p.getHealth() / 20f, t.getHealth() / 20f,
            (float) (dist / 10f),
            (float) closingSpeed,
            p.getAttackCooldownProgress(0.5f),
            t.hurtTime / 10f, // Critical for W-tapping (i-frames detection)
            t.isOnGround() ? 1f : 0f,
            p.isOnGround() ? 1f : 0f,
            p.isSprinting() ? 1f : 0f,
            p.isSubmergedInWater() ? 1f : 0f,
            p.horizontalCollision ? 1f : 0f,
            (float) (p.fallDistance / 5.0),
            p.handSwinging ? 1f : 0f,
            (float) (t.fallDistance / 5.0),
            (float) Math.sqrt(tV.x * tV.x + tV.z * tV.z),
            t.isSprinting() ? 1f : 0f,
            (float) playerLookDot,
            p.isUsingItem() ? 1f : 0f
        };
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] mouseDeltas, int[] keys) {
        // Bound network outputs to [-1, 1] with tanh and scale to a maximum of 3.0 degrees per tick
        float maxDegreesPerTick = 3.0f;
        float pitchDelta = (float) Math.tanh(mouseDeltas[0]) * maxDegreesPerTick;
        float yawDelta   = (float) Math.tanh(mouseDeltas[1]) * maxDegreesPerTick;

        // Apply clamped camera rotations
        player.setPitch(MathHelper.clamp(player.getPitch() + pitchDelta, -90f, 90f));
        player.setYaw(player.getYaw() + yawDelta);

        // Curriculum Masking
        if (NewGen6RLMod.allowMovement) {
            // Stage 2: Full Movement Allowed
            client.options.forwardKey.setPressed(keys[0] == 1);
            client.options.backKey.setPressed(keys[1] == 1);
            client.options.leftKey.setPressed(keys[2] == 1);
            client.options.rightKey.setPressed(keys[3] == 1);
            client.options.jumpKey.setPressed(keys[4] == 1);
            client.options.sprintKey.setPressed(keys[5] == 1);
        } else {
            // Stage 1: Movement Disabled (Force release movement keys)
            client.options.forwardKey.setPressed(false);
            client.options.backKey.setPressed(false);
            client.options.leftKey.setPressed(false);
            client.options.rightKey.setPressed(false);
            client.options.jumpKey.setPressed(false);
            client.options.sprintKey.setPressed(false);
        }

        // Attacking is always allowed
        client.options.attackKey.setPressed(keys[6] == 1);
    }

    @Unique
    private float calculateTacticalReward(ClientPlayerEntity player, LivingEntity target) {
        float reward = 0f;

        // Reset target health baselines on target change
        if (lastTargetUuid == null || !lastTargetUuid.equals(target.getUuid())) {
            lastTargetUuid = target.getUuid();
            prevTargetHealth = target.getHealth();
            prevSelfHealth = player.getHealth();
            return 0f;
        }

        float currentDist = player.distanceTo(target);

        // Damage Dealt Reward
        if (target.getHealth() < prevTargetHealth) {
            float damageDealt = prevTargetHealth - target.getHealth();
            reward += damageDealt * 8.0f;

            // Critical Hit Bonus (landing hits while falling)
            if (player.fallDistance > 0 && !player.isOnGround()) {
                reward += 3.0f;
            }

            // Sprint-Reset Knockback Reward
            if (player.isSprinting()) {
                reward += 2.0f;
            }
        }

        // Damage Taken Penalty
        if (player.getHealth() < prevSelfHealth) {
            reward -= (prevSelfHealth - player.getHealth()) * 6.0f;
        }

        // Sword Reach Spacing Reward (2.0 - 3.2 blocks)
        if (currentDist >= 2.0f && currentDist <= 3.2f) {
            reward += 0.2f;
        } else if (currentDist > 4.5f) {
            reward -= 0.1f;
        }

        // Crosshair Aim Alignment Bonus
        Vec3d lookDir = player.getRotationVector();
        Vec3d toTarget = new Vec3d(target.getX() - player.getX(), target.getEyeY() - player.getEyeY(), target.getZ() - player.getZ()).normalize();
        double dot = lookDir.dotProduct(toTarget);
        if (dot > 0.95) {
            reward += 0.1f;
        }

        // Penalty for attack spamming during target i-frames
        if (target.hurtTime > 0 && player.getAttackCooldownProgress(0.5f) < 0.8f) {
            reward -= 0.15f;
        }

        prevTargetHealth = target.getHealth();
        prevSelfHealth = player.getHealth();
        return reward;
    }

    @Unique
    private LivingEntity getNearestTarget(ClientPlayerEntity player, MinecraftClient client) {
        return (LivingEntity) StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof LivingEntity && e != player && e.isAlive())
            .min(Comparator.comparingDouble(player::distanceTo))
            .orElse(null);
    }
}
