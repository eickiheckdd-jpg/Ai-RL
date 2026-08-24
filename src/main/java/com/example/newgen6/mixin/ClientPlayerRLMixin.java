package com.example.newgen6.mixin;

import com.example.newgen6.NewGen6RLMod;
import com.example.newgen6.rl.PPOAgent.StepData;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.PlayerInput;
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
import java.util.concurrent.CompletableFuture;
import java.util.stream.StreamSupport;

@Mixin(ClientPlayerEntity.class)
public abstract class ClientPlayerRLMixin {

    @Unique private final List<StepData> rlMemory = new ArrayList<>();
    @Unique private LivingEntity lockedTarget = null;
    @Unique private boolean isTraining = false;

    @Inject(method = "tick", at = @At("HEAD"))
    private void onClientTick(CallbackInfo ci) {
        if (!NewGen6RLMod.aiEnabled) return;

        ClientPlayerEntity player = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();

        if (client.world == null || client.isPaused() || player.isDead()) {
            rlMemory.clear();
            return;
        }

        updateTargetLock(player, client);
        if (lockedTarget == null) return;

        float[] state = extract30DState(player, lockedTarget);

        float[] continuousActions = new float[2];
        int[] discreteActions = new int[7]; 
        float[] logProb = new float[1];
        float[] value = new float[1];

        NewGen6RLMod.AGENT.selectAction(state, continuousActions, discreteActions, logProb, value);

        applyInputsDirect(player, client, continuousActions, discreteActions);

        float reward = calculateTacticalReward(player, lockedTarget, discreteActions);
        boolean done = lockedTarget.isDead() || player.isDead();

        rlMemory.add(new StepData(state.clone(), continuousActions[0], continuousActions[1], discreteActions.clone(), logProb[0], reward, value[0], done));

        NewGen6RLMod.lastReward = reward;
        NewGen6RLMod.currentStdPitch = (float) Math.exp(NewGen6RLMod.AGENT.getLogStd()[0]);
        NewGen6RLMod.currentStdYaw = (float) Math.exp(NewGen6RLMod.AGENT.getLogStd()[1]);
        NewGen6RLMod.trainingStep = NewGen6RLMod.AGENT.getTimeStep();

        if ((rlMemory.size() >= 128 || done) && !isTraining) {
            isTraining = true;
            List<StepData> memoryToTrain = new ArrayList<>(rlMemory);
            rlMemory.clear();

            CompletableFuture.runAsync(() -> {
                try {
                    NewGen6RLMod.AGENT.train(memoryToTrain);
                } finally {
                    isTraining = false;
                }
            });
        }
    }

    @Unique
    private void updateTargetLock(ClientPlayerEntity player, MinecraftClient client) {
        if (lockedTarget != null && lockedTarget.isAlive() && player.distanceTo(lockedTarget) <= 16.0f) {
            return; 
        }
        lockedTarget = (LivingEntity) StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof LivingEntity && e != player && e.isAlive() && player.distanceTo(e) <= 16.0f)
            .min(Comparator.comparingDouble(player::distanceTo))
            .orElse(null);
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
            MathHelper.clamp((float) (pV.x * 5.0f), -1f, 1f),
            MathHelper.clamp((float) (pV.y * 2.0f), -1f, 1f),
            MathHelper.clamp((float) (pV.z * 5.0f), -1f, 1f),
            MathHelper.clamp((float) (dx / 16.0), -1f, 1f),
            MathHelper.clamp((float) (dy / 16.0), -1f, 1f),
            MathHelper.clamp((float) (dz / 16.0), -1f, 1f),
            MathHelper.clamp((float) (tV.x * 5.0f), -1f, 1f),
            MathHelper.clamp((float) (tV.y * 2.0f), -1f, 1f),
            MathHelper.clamp((float) (tV.z * 5.0f), -1f, 1f),
            yawDiff / 180f, 
            pitchDiff / 90f,
            (float) targetFacingDot,
            p.getHealth() / 20f, 
            t.getHealth() / 20f,
            MathHelper.clamp((float) (dist / 16.0), 0f, 1f),
            MathHelper.clamp((float) (closingSpeed / 5.0), -1f, 1f),
            p.getAttackCooldownProgress(0.0f),
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
    private void applyInputsDirect(ClientPlayerEntity player, MinecraftClient client, float[] mouseDeltas, int[] keys) {
        if (Float.isNaN(mouseDeltas[0]) || Float.isNaN(mouseDeltas[1])) return;

        float maxDegreesPerTick = 45.0f; 
        float pitchDelta = (float) Math.tanh(mouseDeltas[0]) * maxDegreesPerTick;
        float yawDelta   = (float) Math.tanh(mouseDeltas[1]) * maxDegreesPerTick;

        player.setPitch(MathHelper.clamp(player.getPitch() + pitchDelta, -90f, 90f));
        player.setYaw(player.getYaw() + yawDelta);

        if (NewGen6RLMod.allowMovement) {
            boolean forward = keys[0] == 1;
            boolean back = keys[1] == 1;
            boolean left = keys[2] == 1;
            boolean right = keys[3] == 1;
            boolean jump = keys[4] == 1;
            boolean sneak = false;
            boolean sprint = keys[5] == 1;

            player.input.playerInput = new PlayerInput(forward, back, left, right, jump, sneak, sprint);
            player.setSprinting(sprint);
        }

        if (keys[6] == 1 && player.getAttackCooldownProgress(0.0f) >= 0.9f) {
            client.interactionManager.attackEntity(player, lockedTarget);
            player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
        }
    }

    @Unique
    private float calculateTacticalReward(ClientPlayerEntity player, LivingEntity target, int[] discreteActions) {
        Vec3d lookDir = player.getRotationVector();
        Vec3d toTarget = new Vec3d(
            target.getX() - player.getX(), 
            target.getEyeY() - player.getEyeY(), 
            target.getZ() - player.getZ()
        ).normalize();
        float alignment = (float) lookDir.dotProduct(toTarget);

        float reward = 0f;

        if (alignment > 0.0f) {
            reward += (float) Math.pow(alignment, 4.0) * 0.4f;
        } else {
            reward -= 0.05f; 
        }

        boolean isAttacking = (discreteActions[6] == 1);
        float cooldown = player.getAttackCooldownProgress(0.0f);

        if (isAttacking) {
            if (alignment > 0.85f && cooldown >= 0.9f) {
                reward += 2.0f; 
            } else if (cooldown < 0.8f) {
                reward -= 0.5f; 
            }
        }

        return reward;
    }
}
