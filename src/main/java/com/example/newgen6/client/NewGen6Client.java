package com.example.newgen6.client;

import com.example.newgen6.rl.Observation;
import com.example.newgen6.rl.PolicyNetwork;
import com.example.newgen6.rl.RolloutBuffer;
import com.example.newgen6.rl.PPOTrainer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;

public class NewGen6Client implements ClientModInitializer {

    private final Observation obs = new Observation();
    // Input: 229, Hidden: 64, Output: 2 (Yaw Delta, Pitch Delta)
    private final PolicyNetwork policy = new PolicyNetwork(229, 64, 2); 
    private final RolloutBuffer buffer = new RolloutBuffer(2048, 229, 2);
    private final PPOTrainer trainer = new PPOTrainer(policy);

    private Entity targetEntity = null; // Set this via hotkey or auto-target logic

    @Override
    public void onInitializeClient() {
        // Register the client tick hook using Fabric API
        ClientTickEvents.END_CLIENT_TICK.register(this::onEndTick);
    }

    private void onEndTick(MinecraftClient client) {
        if (client.player == null || client.world == null || targetEntity == null) return;

        // 1. Extract 229-Float Observation
        float[] state = obs.extract(client.player, targetEntity);
        
        // 2. Sample Continuous Action (Yaw and Pitch adjustments)
        float[] action = policy.sampleAction(state);
        
        // 3. Apply Actions to Minecraft Physics (Mapped to mouse GCD logic)
        float maxRotationSpeed = 15.0f; // Max degrees per tick
        float yawDelta = action[0] * maxRotationSpeed;
        float pitchDelta = action[1] * maxRotationSpeed;
        
        client.player.setYaw(client.player.getYaw() + yawDelta);
        client.player.setPitch(client.player.getPitch() + pitchDelta);

        // 4. Calculate Dense Reward (e.g., getting aim error closer to 0)
        float currentYawError = Math.abs(obs.getYawErrorDegrees());
        float reward = (currentYawError < 5.0f) ? 1.0f : -0.1f;
        boolean done = !targetEntity.isAlive();

        // 5. Store Transition in Memory
        // Note: 'value' and 'logProb' are simplified to 0.0f here for the vertical slice
        buffer.store(state, action, reward, 0.0f, 0.0f, done);

        // 6. Train if buffer is full or episode ends
        if (done || buffer.size >= buffer.capacity) {
            trainer.train(buffer);
        }
    }
}
