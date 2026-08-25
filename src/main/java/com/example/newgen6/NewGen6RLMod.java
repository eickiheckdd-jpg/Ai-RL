package com.example.newgen6;

import com.example.newgen6.hud.TrainingState;
import com.example.newgen6.rl.RLAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;

/**
 * Main client entrypoint declared by fabric.mod.json.
 *
 * The initial integration is intentionally conservative:
 * - creates the Java-only RL agent
 * - keeps RL state on the client
 * - does not assume undocumented Minecraft APIs
 * - leaves environment observation/action application to dedicated layers
 */
public final class NewGen6RLMod implements ClientModInitializer {

    private static RLAgent agent;
    private static volatile TrainingState trainingState =
            TrainingState.empty();

    @Override
    public void onInitializeClient() {
        agent = new RLAgent(new java.util.Random());

        ClientTickEvents.END_CLIENT_TICK.register(
                NewGen6RLMod::onClientTick
        );
    }

    private static void onClientTick(MinecraftClient client) {
        if (client == null) {
            return;
        }

        // Integration is deliberately kept separate from Minecraft API
        // assumptions. The environment/mixin layer will feed observations.
    }

    public static RLAgent agent() {
        if (agent == null) {
            throw new IllegalStateException(
                    "NewGen6RLMod has not been initialized"
            );
        }
        return agent;
    }

    public static TrainingState trainingState() {
        return trainingState;
    }

    public static void setTrainingState(
            TrainingState state) {

        if (state == null) {
            throw new IllegalArgumentException(
                    "state cannot be null"
            );
        }

        trainingState = state;
    }
}