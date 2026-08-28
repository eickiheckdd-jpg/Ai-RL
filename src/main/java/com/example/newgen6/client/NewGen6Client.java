package com.example.newgen6.client;

import com.example.newgen6.hud.NewGen6Hud;
import com.example.newgen6.rl.AgentController;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;

import net.minecraft.client.MinecraftClient;

import org.lwjgl.glfw.GLFW;

public class NewGen6Client implements ClientModInitializer {
    private static AgentController controller;

    private static boolean prevC = false;
    private static boolean prevX = false;

    @Override
    public void onInitializeClient() {
        controller = new AgentController();

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            handleToggles(client);
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

    private static void handleToggles(MinecraftClient client) {
        if (client == null) return;

        try {
            long window = client.getWindow().getHandle();
            if (window == 0L) return;

            boolean c = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_C) == GLFW.GLFW_PRESS;
            boolean x = GLFW.glfwGetKey(window, GLFW.GLFW_KEY_X) == GLFW.GLFW_PRESS;

            if (client.currentScreen == null) {
                if (c && !prevC) {
                    controller.toggleAi();
                }

                if (x && !prevX) {
                    controller.toggleHud();
                }
            }

            prevC = c;
            prevX = x;
        } catch (Throwable ignored) {
        }
    }
}