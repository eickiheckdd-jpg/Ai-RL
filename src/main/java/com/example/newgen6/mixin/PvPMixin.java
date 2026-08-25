package com.example.newgen6.mixin;

import com.example.newgen6.NewGen6RLMod;
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
    @Unique private final float[] actions = new float[7];
    @Unique private int saveTimer = 0;

    @Inject(method = "tick", at = @At("HEAD"))
    private void onTick(CallbackInfo ci) {
        ClientPlayerEntity p = (ClientPlayerEntity) (Object) this;
        MinecraftClient client = MinecraftClient.getInstance();

        if (client.world == null || p.isDead() || p.isRemoved()) return;

        // Process C and X keys
        NewGen6RLMod.checkToggles(client);

        if (!NewGen6RLMod.aiActive) return;

        PlayerEntity t = StreamSupport.stream(client.world.getEntities().spliterator(), false)
            .filter(e -> e instanceof PlayerEntity && e != p && e.isAlive() && !e.isRemoved() && p.distanceTo(e) <= 16.0f)
            .map(e -> (PlayerEntity) e)
            .min(Comparator.comparingDouble(p::distanceTo))
            .orElse(null);

        if (t == null) return;

        extractState(p, t, state);
        AGENT.selectAction(state, actions);

        // Update HUD Metrics
        System.arraycopy(actions, 0, NewGen6RLMod.lastActions, 0, actions.length);
        NewGen6RLMod.currentNoise = AGENT.stdev;

        // Apply Aim Inputs
        p.setYaw(p.getYaw() + (actions[0] * 18.0f)); 
        p.setPitch(Math.max(-90.0f, Math.min(90.0f, p.getPitch() + (actions[1] * 18.0f)))); 

        // Inject Movement Inputs
        p.input.playerInput = new PlayerInput(
            actions[2] > 0.0f,   
            actions[2] < -0.3f,  
            actions[5] > 0.2f,   
            actions[6] > 0.2f,   
            actions[3] > 0.5f,   
            false,               
            actions[2] > 0.5f    
        );
        p.setSprinting(actions[2] > 0.5f);

        boolean isAttacking = actions[4] > 0.2f;
        client.options.attackKey.setPressed(isAttacking); 

        extractState(p, t, nextState);
        float reward = calculateReward(p, t, isAttacking);
        NewGen6RLMod.lastReward = reward;

        AGENT.trainAsync(state, actions, reward, nextState);

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
        out[8] = (float) relVel.x;
        out[9] = (float) relVel.y;
        out[10] = (float) relVel.z;
        out[11] = t.hurtTime / 10.0f;
        out[12] = p.hurtTime / 10.0f;
    }

    @Unique
    private float calculateReward(ClientPlayerEntity p, PlayerEntity t, boolean clicked) {
        float r = 0.0f;
        float cd = p.getAttackCooldownProgress(0.0f);
        double dist = p.distanceTo(t);

        // 1. Crosshair Alignment 
        Vec3d toT = t.getEyePos().subtract(p.getEyePos()).normalize();
        double lookAlignment = p.getRotationVector().dotProduct(toT);
        
        if (lookAlignment > 0.85) r += 0.05f; 

        // 2. Click Handling (Patched Exploit)
        if (clicked) {
            if (dist > 3.2f) {
                r -= 1.0f; // Penalty for swinging out of range
            } else if (lookAlignment < 0.85) {
                r -= 1.0f; // Penalty for swinging while looking at the sky/floor
            } else if (cd < 0.85f) {
                r -= 0.5f; // Penalty for spam clicking on cooldown
            } else if (t.hurtTime > 0) {
                r -= 0.3f; // Penalty for hitting during immunity frames
            } else {
                r += 1.0f; // Good attack attempt (Close AND Aiming properly)
            }
        }

        // 3. Impact Rewards
        if (t.hurtTime == 10) r += 15.0f; 
        if (p.hurtTime == 10) r -= 8.0f;  

        return r;
    }
}
