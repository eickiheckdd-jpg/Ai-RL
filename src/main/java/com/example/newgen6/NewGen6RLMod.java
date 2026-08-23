package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NewGen6RLMod implements ModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("newgen6");
    private static int tickCounter = 0;

    @Override
    public void onInitialize() {
        LOGGER.info("Initializing NewGen6 Mobile RL Agent...");
        KeyInputHandler.register();
        
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (client.player == null || !RLConfig.agentEnabled) return;
            
            tickCounter++;
            if (tickCounter % RLConfig.TICK_INTERVAL != 0) return; // Saves mobile CPU

            // AI aiming logic triggers here every 2 ticks
            // e.g., double[] action = PPOAgent.getInstance().getAction(observations);
            // client.player.setYaw(client.player.getYaw() + (float) action[0]);
        });
    }
}