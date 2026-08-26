package com.example.newgen6.client;

import com.example.newgen6.rl.Observation;
import com.example.newgen6.rl.PolicyNetwork;
import com.example.newgen6.rl.RolloutBuffer;
import com.example.newgen6.rl.PPOTrainer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import org.lwjgl.glfw.GLFW;

public class NewGen6Client implements ClientModInitializer {

    private final Observation obs = new Observation();
    private final PolicyNetwork policy = new PolicyNetwork(229, 64, 2); 
    private final RolloutBuffer buffer = new RolloutBuffer(1024, 229, 2);
    private final PPOTrainer trainer = new PPOTrainer(policy);

    // Keybindings & State
    private KeyBinding toggleAiKey;
    private KeyBinding toggleHudKey;
    private boolean aiActive = false;
    private boolean hudVisible = true;
    private float lastReward = 0.0f;

    @Override
    public void onInitializeClient() {
        // Register C key for AI Toggle / ALL STOP
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                "category.newgen6.ai"
        ));

        // Register X key for HUD Overlay Toggle
        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                "category.newgen6.ai"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(this::onEndTick);
        
        // HUD Overlay Renderer
        HudRenderCallback.EVENT.register((drawContext, tickCounter) -> {
            if (!hudVisible) return;
            String status = aiActive ? "§aAI ACTIVE (C to Stop)" : "§cAI STOPPED (C to Start)";
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, status, 10, 10, 0xFFFFFF, true);
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, "Reward: " + lastReward, 10, 25, 0xAAAAAA, true);
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, "Memory Buffer: " + buffer.size + "/" + buffer.capacity, 10, 40, 0xAAAAAA, true);
        });
    }

    private void onEndTick(MinecraftClient client) {
        // 1. Process Key Presses
        while (toggleAiKey.wasPressed()) {
            aiActive = !aiActive;
            if (!aiActive) {
                // ==========================================
                // ALL STOP SAFETY PROTOCOL
                // ==========================================
                // Instantly halts all AI action and resets camera momentum
                if (client.player != null) {
                    client.options.forwardKey.setPressed(false);
                    client.options.attackKey.setPressed(false);
                }
            }
        }
        
        while (toggleHudKey.wasPressed()) {
            hudVisible = !hudVisible;
        }

        // If AI is off, or game world is invalid, skip AI logic entirely
        if (!aiActive || client.player == null || client.world == null) return;

        // Auto-target closest player (simplified target logic)
        Entity target = findClosestTarget(client);
        if (target == null) return;

        // 2. Extract State & Sample AI Action
        float[] state = obs.extract(client.player, target);
        float[] action = policy.sampleAction(state);

        // 3. Apply Actions to Camera (Bounded by max rotation limits)
        float maxSpeed = 15.0f; 
        float yawDelta = action[0] * maxSpeed;
        float pitchDelta = action[1] * maxSpeed;
        
        client.player.setYaw(client.player.getYaw() + yawDelta);
        client.player.setPitch(client.player.getPitch() + pitchDelta);

        // 4. Calculate Rewards & Store
        float currentYawError = Math.abs(obs.getYawErrorDegrees());
        lastReward = (currentYawError < 5.0f) ? 1.0f : -0.1f;
        boolean done = !target.isAlive();

        buffer.store(state, action, lastReward, 0.0f, 0.0f, done);

        // 5. Train Policy when memory buffer is full
        if (done || buffer.size >= buffer.capacity) {
            trainer.train(buffer);
        }
    }

    private Entity findClosestTarget(MinecraftClient client) {
        Entity closest = null;
        double minDistance = 100.0;
        for (Entity entity : client.world.getEntities()) {
            if (entity instanceof PlayerEntity && entity != client.player) {
                double dist = client.player.distanceTo(entity);
                if (dist < minDistance) {
                    minDistance = dist;
                    closest = entity;
                }
            }
        }
        return closest;
    }
}
