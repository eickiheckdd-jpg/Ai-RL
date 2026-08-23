package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.Vec3d;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NewGen6RLMod implements ModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("newgen6");
    private static int tickCounter = 0;

    @Override
    public void onInitialize() {
        LOGGER.info("Starting Mobile RL Agent...");

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (client.player == null || !RLConfig.agentEnabled) return;

            tickCounter++;
            if (tickCounter % RLConfig.TICK_INTERVAL != 0) return; // Tick-skipping for mobile CPU

            // 1. Calculate simplified observation (Looking for closest target/mob)
            float[] obs = getPlayerObservations(client);

            // 2. Predict actions with lightweight DL4J inference
            float[] action = PPOAgent.getInstance().predict(obs);

            // 3. Apply smooth camera rotation
            client.player.setYaw(client.player.getYaw() + action[0]);
            client.player.setPitch(client.player.getPitch() + action[1]);
        });
    }

    private float[] getPlayerObservations(MinecraftClient client) {
        float[] obs = new float[RLConfig.OBS_DIM];
        if (client.player == null) return obs;

        Vec3d pPos = client.player.getPos();
        Entity target = null;
        double closestDist = 30.0;

        // Find nearest entity in 30 block radius
        if (client.world != null) {
            for (Entity e : client.world.getEntities()) {
                if (e != client.player && e.isAlive()) {
                    double dist = pPos.distanceTo(e.getPos());
                    if (dist < closestDist) {
                        closestDist = dist;
                        target = e;
                    }
                }
            }
        }

        if (target != null) {
            Vec3d tPos = target.getPos();
            obs[0] = (float) (tPos.x - pPos.x);
            obs[1] = (float) (tPos.y - pPos.y);
            obs[2] = (float) (tPos.z - pPos.z);
            obs[3] = client.player.getPitch();
            obs[4] = client.player.getYaw();
            obs[5] = (float) closestDist;
        }

        return obs;
    }
}
