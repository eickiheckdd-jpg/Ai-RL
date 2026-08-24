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
        // Only run if the player has toggled the AI ON
        if (!NewGen6RLMod.aiEnabled) return;

        ClientPlayerEntity player = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.world == null || player.isDead()) return;

        LivingEntity target = getNearestTarget(player, client);
        if (target == null) return; 

        // 1. Gather 14D State
        float[] state = extractState(player, target);

        // 2. Network Inference (Continuous & Discrete)
        float[] continuousActions = new float[2]; // Pitch, Yaw Deltas
        int[] discreteActions = new int[3];       // Move, Jump, Attack
        float[] logProb = new float[1];
        float[] value = new float[1];
        
        NewGen6RLMod.AGENT.selectAction(state, continuousActions, discreteActions, logProb, value);

        // 3. Apply Continuous Mouse & Discrete Key Inputs
        applyInputs(player, client, continuousActions, discreteActions);

        // 4. Calculate Reward
        float reward = calculateReward(player, target);
        boolean done = target.isDead() || player.isDead();
        
        rlMemory.add(new StepData(state, continuousActions[0], continuousActions[1], discreteActions, logProb[0], reward, value[0], done));

        if (rlMemory.size() >= 128 || done) {
            NewGen6RLMod.AGENT.train(rlMemory);
            rlMemory.clear();
        }
    }

    @Unique
    private float[] extractState(ClientPlayerEntity p, LivingEntity t) {
        Vec3d pV = p.getVelocity();
        Vec3d tV = t.getVelocity();
        return new float[] {
            (float) pV.x, (float) pV.y, (float) pV.z,
            (float) (t.getX() - p.getX()), (float) (t.getY() - p.getY()), (float) (t.getZ() - p.getZ()),
            p.getHealth(), p.getAttackCooldownProgress(0.5f), p.distanceTo(t),
            (float) tV.x, (float) tV.y, (float) tV.z, t.getHealth(), p.isSprinting() ? 1f : 0f
        };
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] mouseDeltas, int[] keys) {
        // Continuous Mouse Deltas applied directly
        player.setPitch(MathHelper.clamp(player.getPitch() + mouseDeltas[0], -90f, 90f));
        player.setYaw(player.getYaw() + mouseDeltas[1]);

        // Movement (0=Idle, 1=W, 2=S, 3=A, 4=D)
        client.options.forwardKey.setPressed(keys[0] == 1);
        client.options.backKey.setPressed(keys[0] == 2);
        client.options.leftKey.setPressed(keys[0] == 3);
        client.options.rightKey.setPressed(keys[0] == 4);

        client.options.jumpKey.setPressed(keys[1] == 1);
        
        if (keys[2] == 1 && player.getAttackCooldownProgress(0.5f) > 0.9f) {
            client.options.attackKey.setPressed(true); 
        } else {
            client.options.attackKey.setPressed(false);
        }
    }

    @Unique
    private float calculateReward(ClientPlayerEntity player, LivingEntity target) {
        float reward = 0f;
        if (target.getHealth() < prevTargetHealth) reward += (prevTargetHealth - target.getHealth()) * 5.0f;
        if (player.getHealth() < prevSelfHealth) reward -= (prevSelfHealth - player.getHealth()) * 3.0f;
        float dist = player.distanceTo(target);
        if (dist > 3.0f) reward -= 0.1f;
        if (dist < 1.0f) reward -= 0.1f;
        
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