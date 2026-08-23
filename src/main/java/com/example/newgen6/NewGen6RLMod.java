package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NewGen6RLMod implements ClientModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("newgen6");
    private static int tickCounter = 0;

    @Override
    public void onInitializeClient() {
        LOGGER.info("Starting Mobile RL Agent...");
        
        // This makes your C and X keys work!
        KeyInputHandler.register();

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (client.player == null || !RLConfig.agentEnabled) return;

            tickCounter++;
            if (tickCounter % RLConfig.TICK_INTERVAL != 0) return;

            float[] obs = getPlayerObservations(client);
            PPOAgent.InferenceResult action = PPOAgent.getInstance().predict(obs);

            CombatActionExecutor.execute(client.player, action);
        });
    }

    private float[] getPlayerObservations(MinecraftClient client) {
        float[] obs = new float[RLConfig.OBS_DIM];
        if (client.player == null) return obs;

        double px = client.player.getX();
        double py = client.player.getY();
        double pz = client.player.getZ();

        Entity target = null;
        double closestDist = 30.0;

        if (client.world != null) {
            for (Entity e : client.world.getEntities()) {
                if (e != client.player && e.isAlive()) {
                    double ex = e.getX();
                    double ey = e.getY();
                    double ez = e.getZ();

                    double dist = Math.sqrt(Math.pow(ex - px, 2) + Math.pow(ey - py, 2) + Math.pow(ez - pz, 2));
                    if (dist < closestDist) {
                        closestDist = dist;
                        target = e;
                    }
                }
            }
        }

        if (target != null) {
            obs[0] = (float) (target.getX() - px);
            obs[1] = (float) (target.getY() - py);
            obs[2] = (float) (target.getZ() - pz);
            obs[3] = client.player.getPitch();
            obs[4] = client.player.getYaw();
            obs[5] = (float) closestDist;
        }

        return obs;
    }
}