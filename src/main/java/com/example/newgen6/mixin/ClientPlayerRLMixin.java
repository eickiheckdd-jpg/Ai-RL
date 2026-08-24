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
        int[] discreteActions = new int[7]; 
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

        Vec3d targetLook = t.getRotationVector();
        Vec3d toPlayer = new Vec3d(-dx, -dy, -dz).normalize();
        double targetFacingDot = targetLook.dotProduct(toPlayer);

        Vec3d playerLook = p.getRotationVector();
        Vec3d toTarget = new Vec3d(dx, dy, dz).normalize();
        double playerLookDot = playerLook.dotProduct(toTarget);

        Vec3d relVel = pV.subtract(tV);
        double closingSpeed = relVel.dotProduct(toTarget);

        return new float[] {
            MathHelper.clamp((float) pV.x, -1f, 1f),
            MathHelper.clamp((float) pV.y, -1f, 1f),
            MathHelper.clamp((float) pV.z, -1f, 1f),
            MathHelper.clamp((float) (dx / 16.0), -1f, 1f),
            MathHelper.clamp((float) (dy / 16.0), -1f, 1f),
            MathHelper.clamp((float) (dz / 16.0), -1f, 1f),
            MathHelper.clamp((float) tV.x, -1f, 1f),
            MathHelper.clamp((float) tV.y, -1f, 1f),
            MathHelper.clamp((float) tV.z, -1f, 1f),
            yawDiff / 180f, 
            pitchDiff / 90f,
            (float) targetFacingDot,
            p.getHealth() / 20f, 
            t.getHealth() / 20f,
            MathHelper.clamp((float) (dist / 16.0), 0f, 1f),
            MathHelper.clamp((float) (closingSpeed / 5.0), -1f, 1f),
            p.getAttackCooldownProgress(0.5f),
            t.hurtTime / 10f,
            t.isOnGround() ? 1f : 0f,
            p.isOnGround() ? 1f : 0f,
            p.isSprinting() ? 1f : 0f,
            p.isSubmergedInWater() ? 1f : 0f,
            p.horizontalCollision ? 1f : 0f,
            MathHelper.clamp((float) (p.fallDistance / 5.0), 0f, 1f),
            p.handSwinging ? 1f : 0f,
            MathHelper.clamp((float) (t.fallDistance / 5.0), 0f, 1f),
            MathHelper.clamp((float) Math.sqrt(tV.x * tV.x + tV.z * tV.z), 0f, 1f),
            t.isSprinting() ? 1f : 0f,
            (float) playerLookDot,
            p.isUsingItem() ? 1f : 0f
        };
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] mouseDeltas, int[] keys) {
        if (Float.isNaN(mouseDeltas[0]) || Float.isNaN(mouseDeltas[1]) ||
            Float.isInfinite(mouseDeltas[0]) || Float.isInfinite(mouseDeltas[1])) {
            return;
        }

        float maxDegreesPerTick = 2.5f;
        float pitchDelta = (float) Math.tanh(mouseDeltas[0]) * maxDegreesPerTick;
        float yawDelta   = (float) Math.tanh(mouseDeltas[1]) * maxDegreesPerTick;

        player.setPitch(MathHelper.clamp(player.getPitch() + pitchDelta, -90f, 90f));
        player.setYaw(player.getYaw() + yawDelta);

        if (NewGen6RLMod.allowMovement) {
            client.options.forwardKey.setPressed(keys[0] == 1);
            client.options.backKey.setPressed(keys[1] == 1);
            client.options.leftKey.setPressed(keys[2] == 1);
            client.options.rightKey.setPressed(keys[3] == 1);
            client.options.jumpKey.setPressed(keys[4] == 1);
            client.options.sprintKey.setPressed(keys[5] == 1);
        } else {
            client.options.forwardKey.setPressed(false);
            client.options.backKey.setPressed(false);
            client.options.leftKey.setPressed(false);
            client.options.rightKey.setPressed(false);
            client.options.jumpKey.setPressed(false);
            client.options.sprintKey.setPressed(false);
        }

        client.options.attackKey.setPressed(keys[6] == 1);
    }

    @Unique
    private float calculateTacticalReward(ClientPlayerEntity player, LivingEntity target) {
        float reward = 0f;

        if (lastTargetUuid == null || !lastTargetUuid.equals(target.getUuid())) {
            lastTargetUuid = target.getUuid();
            prevTargetHealth = target.getHealth();
            prevSelfHealth = player.getHealth();
            return 0f;
        }

        float currentDist = player.distanceTo(target);

        if (target.getHealth() < prevTargetHealth) {
            float damageDealt = prevTargetHealth - target.getHealth();
            reward += damageDealt * 8.0f;

            if (player.fallDistance > 0 && !player.isOnGround()) {
                reward += 3.0f;
            }

            if (player.isSprinting()) {
                reward += 2.0f;
            }
        }

        if (player.getHealth() < prevSelfHealth) {
            reward -= (prevSelfHealth - player.getHealth()) * 6.0f;
        }

        if (currentDist >= 2.0f && currentDist <= 3.2f) {
            reward += 0.2f;
        } else if (currentDist > 4.5f) {
            reward -= 0.1f;
        }

        Vec3d lookDir = player.getRotationVector();
        Vec3d toTarget = new Vec3d(target.getX() - player.getX(), target.getEyeY() - player.getEyeY(), target.getZ() - player.getZ()).normalize();
        double dot = lookDir.dotProduct(toTarget);
        if (dot > 0.95) {
            reward += 0.1f;
        }

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
