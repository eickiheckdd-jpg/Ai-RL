package com.example.newgen6;

import com.example.newgen6.client.RLClientRuntime;
import com.example.newgen6.hud.TrainingState;
import com.example.newgen6.rl.RLAgent;
import net.fabricmc.api.ClientModInitializer;

/**
 * Main client entrypoint declared by fabric.mod.json.
 */
public final class NewGen6RLMod implements ClientModInitializer {

    private static RLAgent agent;
    private static volatile TrainingState trainingState =
            TrainingState.empty();

    @Override
    public void onInitializeClient() {
        agent = new RLAgent(new java.util.Random());

        // Initializes:
        // - C = AI/training toggle
        // - X = HUD toggle
        // - RL training runtime
        // - live training HUD
        RLClientRuntime.initialize();
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