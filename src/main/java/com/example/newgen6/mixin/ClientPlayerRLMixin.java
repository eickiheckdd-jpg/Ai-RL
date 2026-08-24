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

@Mixin(ClientPlayerEntity.class)
public abstract class ClientPlayerRLMixin {

    @Unique private final List<StepData> rlMemory = new ArrayList<>();
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

        // 1. Gather Advanced 21-Dimensional State Vector
        float[] state = extract21DState(player, target);

        // 2. Action Inference
        float[] continuousActions = new float[2];
        int[] discreteActions = new int[3];       
        float[] logProb = new float[1];
        float[] value = new float[1];
        
        NewGen6RLMod.AGENT.selectAction(state, continuousActions, discreteActions, logProb, value);

        // 3. Execution
        applyInputs(player, client, continuousActions, discreteActions);

        // 4. Calculate Enhanced Reward
        float reward = calculateTacticalReward(player, target);
        boolean done = target.isDead() || player.isDead();
        
        rlMemory.add(new StepData(state, continuousActions[0], continuousActions[1], discreteActions, logProb[0], reward, value[0], done));

        if (rlMemory.size() >= 128 || done) {
            NewGen6RLMod.AGENT.train(rlMemory);
            rlMemory.clear();
        }
    }

    @Unique
    private float[] extract21DState(ClientPlayerEntity p, LivingEntity t) {
        Vec3d pV = p.getVelocity();
        Vec3d tV = t.getVelocity();
        
        // Aim Alignment Calculations
        double dx = t.getX() - p.getX();
        double dy = (t.getY() + t.getStandingEyeHeight()) - (p.getY() + p.getStandingEyeHeight());
        double dz = t.getZ() - p.getZ();
        double distXZ = Math.sqrt(dx * dx + dz * dz);
        
        float targetYaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0f);
        float targetPitch = (float) (-Math.toDegrees(Math.atan2(dy, distXZ)));
        
        float yawDiff = MathHelper.wrapDegrees(targetYaw - p.getYaw());
        float pitchDiff = MathHelper.wrapDegrees(targetPitch - p.getPitch());

        return new float[] {
            // Player Motion (3)
            (float) pV.x, (float) pV.y, (float) pV.z,
            // Relative Target Position (3)
            (float) dx, (float) dy, (float) dz,
            // Target Motion (3)
            (float) tV.x, (float) tV.y, (float) tV.z,
            // Crosshair Alignment Offsets (2)
            yawDiff / 180f, pitchDiff / 90f,
            // Healths & Distance (3)
            p.getHealth() / 20f, t.getHealth() / 20f, p.distanceTo(t) / 10f,
            // Combat Mechanics (4)
            p.getAttackCooldownProgress(0.5f),
            t.hurtTime > 0 ? 1f : 0f, // Target invulnerability ticks
            p.isOnGround() ? 1f : 0f,
            p.isSprinting() ? 1f : 0f,
            // Environment Context (3)
            p.isSubmergedInWater() ? 1f : 0f,
            p.horizontalCollision ? 1f : 0f, // Stuck against a wall
            p.fallDistance > 0 ? 1f : 0f      // In air falling (Potential Critical Hit!)
        };
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] mouseDeltas, int[] keys) {
        player.setPitch(MathHelper.clamp(player.getPitch() + mouseDeltas[0], -90f, 90f));
        player.setYaw(player.getYaw() + mouseDeltas[1]);

        client.options.forwardKey.setPressed(keys[0] == 1);
        client.options.backKey.setPressed(keys[0] == 2);
        client.options.leftKey.setPressed(keys[0] == 3);
        client.options.rightKey.setPressed(keys[0] == 4);

        client.options.jumpKey.setPressed(keys[1] == 1);
        
        if (keys[2] == 1 && player.getAttackCooldownProgress(0.5f) > 0.85f) {
            client.options.attackKey.setPressed(true); 
        } else {
            client.options.attackKey.setPressed(false);
        }
    }

    @Unique
    private float calculateTacticalReward(ClientPlayerEntity player, LivingEntity target) {
        float reward = 0f;

        // Damage rewards
        if (target.getHealth() < prevTargetHealth) {
            float damageDealt = prevTargetHealth - target.getHealth();
            reward += damageDealt * 6.0f;
            
            // Critical Hit Bonus (hitting while falling)
            if (player.fallDistance > 0 && !player.isOnGround()) {
                reward += 2.5f; 
            }
        }
        
        if (player.getHealth() < prevSelfHealth) {
            reward -= (prevSelfHealth - player.getHealth()) * 4.0f;
        }

        // Spacing & Distance Management
        float dist = player.distanceTo(target);
        if (dist >= 1.8f && dist <= 3.2f) {
            reward += 0.2f; // Optimal Melee Range
        } else if (dist > 5.0f) {
            reward -= 0.3f; // Too far away penalty
        }

        // Crosshair Aim Precision Reward
        Vec3d lookDir = player.getRotationVector();
        Vec3d toTarget = new Vec3d(target.getX() - player.getX(), target.getEyeY() - player.getEyeY(), target.getZ() - player.getZ()).normalize();
        double dotProduct = lookDir.dotProduct(toTarget);
        if (dotProduct > 0.9) {
            reward += 0.15f; // Crosshair is centered on target
        }

        prevTargetHealth = target.getHealth();
        prevSelfHealth = player.getHealth();
        return reward;
    }

    @Unique
    private LivingEntity getNearestTarget(ClientPlayerEntity player, MinecraftClient client) {
        return (LivingEntity) client.world.getEntities().stream()
            .filter(e -> e instanceof LivingEntity && e != player && e.isAlive())
            .min(Comparator.comparingDouble(player::distanceTo))
            .orElse(null);
    }
}
