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
    @Unique private final float[] state = new float[13];
    @Unique private final float[] nextState = new float[13];
    @Unique private final float[] actions = new float[5];
    @Unique private int saveTimer = 0;

    @Inject(method = "tick", at = @At("HEAD"))
    private void onTick(CallbackInfo ci) {
        ClientPlayerEntity p = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();

        // Null Guard & Lifecycle Checks
        if (client.world == null || p.isDead() || p.isRemoved()) return;

        PlayerEntity t = StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof PlayerEntity && e != p && e.isAlive() && !e.isRemoved() && p.distanceTo(e) <= 16.0f)
            .map(e -> (PlayerEntity) e)
            .min(Comparator.comparingDouble(p::distanceTo))
            .orElse(null);

        if (t == null) return;

        extractState(p, t, state);
        AGENT.selectAction(state, actions);

        // Native Continuous Aim Injection
        p.setYaw(p.getYaw() + (actions[0] * 18.0f)); 
        float targetPitch = p.getPitch() + (actions[1] * 18.0f);
        p.setPitch(Math.max(-90.0f, Math.min(90.0f, targetPitch))); 

        // Native Movement & Combat Injection
        p.input.playerInput = new PlayerInput(
            actions[2] > 0.0f,  // W
            actions[2] < -0.3f, // S
            false, false,       // Strafe A/D
            actions[4] > 0.5f,  // Space (Jump)
            false,              // Sneak
            actions[3] > 0.0f   // Sprint
        );
        p.setSprinting(actions[3] > 0.0f);
        client.options.attackKey.setPressed(actions[4] > 0.0f); 

        extractState(p, t, nextState);
        float reward = calculateReward(p, t, actions[4] > 0.0f);
        AGENT.trainAsync(state, actions, reward, nextState);

        // Auto-Save Management
        saveTimer++;
        if (saveTimer >= 6000 || t.isDead() || p.getHealth() <= 0) {
            AGENT.saveBrainAsync();
            saveTimer = 0;
        }
    }

    @Unique
    private void extractState(ClientPlayerEntity p, PlayerEntity t, float[] out) {
        Vec3d toT = t.getEyePos().subtract(p.getEyePos()).normalize();
        Vec3d look = p.getRotationVector();
        Vec3d relVel = t.getVelocity().subtract(p.getVelocity());

        out[0] = (float) (t.getX() - p.getX()) / 16.0f;
        out[1] = (float) (t.getY() - p.getY()) / 16.0f;
        out[2] = (float) (t.getZ() - p.getZ()) / 16.0f;
        out[3] = (float) look.dotProduct(toT);      
        out[4] = p.getHealth() / 20.0f;
        out[5] = t.getHealth() / 20.0f;
        out[6] = p.getAttackCooldownProgress(0.0f); 
        out[7] = p.distanceTo(t) / 16.0f;
        out[8] = (float) relVel.x;                  // Velocity X
        out[9] = (float) relVel.y;                  // Velocity Y
        out[10] = (float) relVel.z;                 // Velocity Z
        out[11] = t.hurtTime / 10.0f;               // Target Invulnerability
        out[12] = p.hurtTime / 10.0f;               // Self Invulnerability
    }

    @Unique
    private float calculateReward(ClientPlayerEntity p, PlayerEntity t, boolean clicked) {
        float r = 0.0f;
        float cd = p.getAttackCooldownProgress(0.0f);
        double dist = p.distanceTo(t);

        if (clicked) {
            if (cd < 0.9f) r -= 0.6f;             // Penalty: Spam clicking
            else if (t.hurtTime > 0) r -= 0.8f;   // Penalty: Hitting during invulnerability frames
            else if (dist <= 3.0f) r += 1.8f;     // Reward: Perfect combo timing
        }
        
        if (t.hurtTime == 10) r += 10.0f;         
        if (p.hurtTime == 10) r -= 5.0f;          
        return r;
    }
}
