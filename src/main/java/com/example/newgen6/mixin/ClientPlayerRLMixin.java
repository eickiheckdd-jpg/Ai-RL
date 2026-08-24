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
    @Unique private final float[] stateBuffer = new float[30];
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
        double dx = t.getX() - p.getX(), dy = t.getEyeY() - p.getEyeY(), dz = t.getZ() - p.getZ();
        out[0] = (float) pV.x; out[1] = (float) pV.y; out[2] = (float) pV.z;
        out[3] = (float) dx; out[4] = (float) dy; out[5] = (float) dz;
        out[6] = p.getHealth(); out[7] = t.getHealth();
        out[8] = p.getAttackCooldownProgress(0.0f);
    }

    @Unique
    private void applyInputs(ClientPlayerEntity player, MinecraftClient client, float[] actions) {
        double mult = Math.pow(client.options.getMouseSensitivity().getValue() * 0.6 + 0.2, 3) * 8.0;
        player.changeLookDirection(actions[1] * mult, actions[0] * mult);

        player.input.playerInput = new PlayerInput(actions[2] > 0, actions[2] < 0, actions[3] < 0, actions[3] > 0, actions[4] > 0, actions[7] > 0, actions[5] > 0);
        player.setSprinting(actions[5] > 0);

        if (actions[6] > 0.5f && isLookingAtBox(player, lockedTarget)) {
            client.interactionManager.attackEntity(player, lockedTarget);
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
