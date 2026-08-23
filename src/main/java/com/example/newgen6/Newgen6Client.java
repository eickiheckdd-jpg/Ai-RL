package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;

public class Newgen6Client implements ClientModInitializer {
    private TrainingHudOverlay hudOverlay;
    private RLClientTickHandler tickHandler;

    @Override
    public void onInitializeClient() {
        System.out.println("[Newgen6] Initializing Pure-Java RL Agent Prototype...");

        hudOverlay = new TrainingHudOverlay();
        tickHandler = new RLClientTickHandler(hudOverlay);

        HudRenderCallback.EVENT.register(hudOverlay);
        ClientTickEvents.END_CLIENT_TICK.register(tickHandler);

        // Guarantee safe thread shutdown and final weight save on exit
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            System.out.println("[Newgen6] Client stopping. Executing safe agent shutdown...");
            if (tickHandler != null) {
                tickHandler.shutdown();
            }
        });
    }
}
