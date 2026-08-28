package com.example.newgen6.client;

import com.example.newgen6.hud.NewGen6Hud;
import com.example.newgen6.rl.AgentController;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;

import org.lwjgl.glfw.GLFW;

public class NewGen6Client implements ClientModInitializer {
    private static AgentController controller;

    @Override
    public void onInitializeClient() {
        controller = new AgentController();

        KeyBinding toggleAi = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                "category.newgen6.rl"
        ));

        KeyBinding toggleHud = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                "category.newgen6.rl"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleAi.wasPressed()) {
                controller.toggleAi();
            }

            while (toggleHud.wasPressed()) {
                controller.toggleHud();
            }

            controller.onClientTick(client);
        });

        HudRenderCallback.EVENT.register((drawContext, renderTickCounter) -> {
            MinecraftClient client = MinecraftClient.getInstance();
            NewGen6Hud.render(client, drawContext, controller);
        });

        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            controller.saveCheckpointNow();
        });
    }
}