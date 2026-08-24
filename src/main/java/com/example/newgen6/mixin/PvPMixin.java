package com.example.newgen6.mixin;

import com.example.newgen6.rl.PPOEngine;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
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
public abstract class PvPMixin {
    @Unique private static final PPOEngine AGENT = new PPOEngine();
    @Unique private final float[] state = new float[8];
    @Unique private final float[] nextState = new float[8];
    @Unique private final float[] actions = new float[5];

    @Inject(method = "tick", at = @At("HEAD"))
    private void onTick(CallbackInfo ci) {
        ClientPlayerEntity p = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.world == null || p.isDead()) return;

        PlayerEntity t = StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof PlayerEntity && e != p && e.isAlive() && p.distanceTo(e) <= 16.0f)
            .map(e -> (PlayerEntity) e).min(Comparator.comparingDouble(p::distanceTo)).orElse(null);

        if (t == null) return;

        extractState(p, t, state);
        AGENT.selectAction(state, actions);

        // Native Camera, Key, and Mouse Inputs
        p.setYaw(p.getYaw() + (actions[0] * 15.0f));
        p.setPitch(Math.max(-90.0f, Math.min(90.0f, p.getPitch() + (actions[1] * 15.0f))));

        p.input.playerInput = new PlayerInput(
            actions[2] > 0.0f,  // W
            actions[2] < -0.3f, // S
            false, false,       // Strafe A/D
            actions[4] > 0.5f,  // Jump
            false,              // Sneak
            actions[3] > 0.0f   // Sprint
        );
        p.setSprinting(actions[3] > 0.0f);
        client.options.attackKey.setPressed(actions[4] > 0.0f); // Native Left-Click Injection

        extractState(p, t, nextState);
        float reward = calculateReward(p, t, actions[4] > 0.0f);
        AGENT.trainAsync(state, actions, reward, nextState);
    }

    @Unique
    private void extractState(ClientPlayerEntity p, PlayerEntity t, float[] out) {
        Vec3d toT = t.getEyePos().subtract(p.getEyePos()).normalize();
        Vec3d look = p.getRotationVector();
        out[0] = (float) (t.getX() - p.getX()) / 16.0f;
        out[1] = (float) (t.getY() - p.getY()) / 16.0f;
        out[2] = (float) (t.getZ() - p.getZ()) / 16.0f;
        out[3] = (float) look.dotProduct(toT);            // Crosshair alignment
        out[4] = p.getHealth() / 20.0f;
        out[5] = t.getHealth() / 20.0f;
        out[6] = p.getAttackCooldownProgress(0.0f);       // 1.9+ Cooldown bar
        out[7] = p.distanceTo(t) / 16.0f;
    }

    @Unique
    private float calculateReward(ClientPlayerEntity p, PlayerEntity t, boolean clicked) {
        float r = 0.0f;
        float cd = p.getAttackCooldownProgress(0.0f);
        double dist = p.distanceTo(t);

        // Natural Anti-Spam Penalties & Patience Rewards
        if (clicked) {
            if (cd < 0.9f) r -= 0.5f; // Spam click penalty
            else if (dist <= 3.0f) r += 1.5f; // Well-timed swing reward
        }
        if (t.hurtTime == 10) r += 10.0f;  // Target damaged
        if (p.hurtTime == 10) r -= 5.0f;   // Player took damage
        return r;
    }
}
