package com.example.newgen6.mixin;

import com.example.newgen6.NewGen6RLMod;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.Hand;
import net.minecraft.util.PlayerInput;
import net.minecraft.util.math.Vec3d;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.Comparator;
import java.util.stream.StreamSupport;

@Mixin(ClientPlayerEntity.class)
public abstract class ClientPlayerRLMixin {
    @Unique private PlayerEntity lockedTarget = null;
    @Unique private final float[] stateBuffer = new float[12];
    @Unique private final float[] actionBuffer = new float[8];

    @Inject(method = "tick", at = @At("HEAD"))
    private void onClientTick(CallbackInfo ci) {
        if (!NewGen6RLMod.aiEnabled) return;
        ClientPlayerEntity player = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();

        if (client.world == null || player.isDead() || player.isRemoved()) return;

        lockedTarget = StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof PlayerEntity && e != player && e.isAlive() && player.distanceTo(e) <= 16.0f)
            .map(e -> (PlayerEntity) e).min(Comparator.comparingDouble(player::distanceTo)).orElse(null);

        if (lockedTarget == null) return;

        extractState(player, lockedTarget, stateBuffer);

        NewGen6RLMod.AGENT.selectAction(stateBuffer, actionBuffer);
        for (float act : actionBuffer) if (Float.isNaN(act)) return;

        applyInputs(player, client, actionBuffer);

        float reward = calculateReward(player, lockedTarget, actionBuffer);
        boolean done = lockedTarget.isDead() || player.isDead();

        NewGen6RLMod.lastReward = reward;
        NewGen6RLMod.AGENT.storeMemoryAndTrain(stateBuffer, actionBuffer, reward, done);
    }

    @Unique
    private void extractState(ClientPlayerEntity p, PlayerEntity t, float[] out) {
        Vec3d pV = p.getVelocity();
        double dx = t.getX() - p.getX();
        double dy = t.getEyeY() - p.getEyeY();
        double dz = t.getZ() - p.getZ();
        Vec3d look = p.getRotationVector();

        // Normalized distances (-1.0 to 1.0 based on 16 block max range)
        out[0] = (float) (dx / 16.0);
        out[1] = (float) (dy / 16.0);
        out[2] = (float) (dz / 16.0);

        // Velocity
        out[3] = (float) pV.x;
        out[4] = (float) pV.y;
        out[5] = (float) pV.z;

        // Player look angles and orientation vectors
        out[6] = (float) look.x;
        out[7] = (float) look.y;
        out[8] = (float) look.z;

        // Health and Attack Cooldowns (Normalized 0.0 to 1.0)
        out[9] = p.getHealth() / 20.0f;
        out[10] = t.getHealth() / 20.0f;
        out[11] = p.getAttackCooldownProgress(0.0f);
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] actions) {
        // Safe, human-like rotation capping
        // actionBuffer elements are [-1.0, 1.0]. Scaling by 3.0 gives smooth 3-degree movements per tick.
        double yawDelta = Math.max(-6.0, Math.min(6.0, actions[0] * 3.0));   // Action 0 = Yaw (Left/Right)
        double pitchDelta = Math.max(-6.0, Math.min(6.0, actions[1] * 3.0)); // Action 1 = Pitch (Up/Down)

        // Correct cursor order: (Yaw Delta, Pitch Delta)
        player.changeLookDirection(yawDelta, pitchDelta);

        // Movement & Actions
        player.input.playerInput = new PlayerInput(
            actions[2] > 0,   // Forward
            actions[2] < 0,   // Back
            actions[3] < 0,   // Left
            actions[3] > 0,   // Right
            actions[4] > 0,   // Jump
            actions[7] > 0,   // Sneak
            actions[5] > 0    // Sprint
        );
        player.setSprinting(actions[5] > 0);

        // Attack Logic
        if (actions[6] > 0.5f && isLookingAtBox(player, lockedTarget)) {
            if (client.interactionManager != null) {
                client.interactionManager.attackEntity(player, lockedTarget);
            }
            player.swingHand(Hand.MAIN_HAND);
        }
    }

    @Unique
    private boolean isLookingAtBox(ClientPlayerEntity player, PlayerEntity target) {
        Vec3d look = player.getRotationVector();
        Vec3d toTarget = target.getBoundingBox().getCenter().subtract(player.getEyePos()).normalize();
        return look.dotProduct(toTarget) > 0.95;
    }

    @Unique
    private float calculateReward(ClientPlayerEntity player, PlayerEntity target, float[] actions) {
        float r = isLookingAtBox(player, target) ? 0.2f : -0.1f;
        if (actions[6] > 0.5f) r += (isLookingAtBox(player, target) && player.getAttackCooldownProgress(0f) >= 0.9f) ? 2f : -1f;
        if (target.hurtTime == 10) r += 5f;
        return r;
    }
}
