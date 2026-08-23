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
            if (tickCounter % RLConfig.TICK_INTERVAL != 0) return;

            float[] obs = getPlayerObservations(client);
            PPOAgent.InferenceResult action = PPOAgent.getInstance().predict(obs);

            client.player.setYaw(client.player.getYaw() + action.yawDelta);
            client.player.setPitch(client.player.getPitch() + action.pitchDelta);
        });
    }

    private float[] getPlayerObservations(MinecraftClient client) {
        float[] obs = new float[RLConfig.OBS_DIM];
        if (client.player == null) return obs;

        Vec3d pPos = client.player.getPos();
        Entity target = null;
        double closestDist = 30.0;

        if (client.world != null) {
            for (Entity e : client.world.getEntities()) {
                if (e != client.player && e.isAlive()) {
                    Vec3d ePos = e.getPos();
                    double dist = pPos.distanceTo(ePos);
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
