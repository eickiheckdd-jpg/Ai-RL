package com.example.newgen6.client;

import com.example.newgen6.hud.TrainingHud;
import com.example.newgen6.hud.TrainingState;
import com.example.newgen6.rl.RLTrainingManager;
import com.example.newgen6.rl.env.Observation;
import com.example.newgen6.rl.env.PvPEnvironment;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

/**
 * Client-side RL runtime.
 *
 * C toggles AI control/training.
 * X toggles the HUD.
 */
public final class RLClientRuntime {

    private static final KeyBinding.Category CATEGORY =
            KeyBinding.Category.create(
                    Identifier.of("newgen6", "controls")
            );

    private static RLClientRuntime instance;

    private final RLTrainingManager training;
    private final PvPEnvironment environment;

    private final KeyBinding toggleAiKey;
    private final KeyBinding toggleHudKey;

    private boolean aiEnabled;
    private boolean hudEnabled = true;

    private long lastTick = Long.MIN_VALUE;

    private RLClientRuntime() {
        java.util.Random random = new java.util.Random();

        training = new RLTrainingManager(random);
        environment = new PvPEnvironment();

        toggleAiKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.toggle_ai",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        CATEGORY
                )
        );

        toggleHudKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.toggle_hud",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_X,
                        CATEGORY
                )
        );
    }

    public static void initialize() {
        if (instance != null) {
            return;
        }

        instance = new RLClientRuntime();

        ClientTickEvents.END_CLIENT_TICK.register(
                RLClientRuntime::tick
        );

        TrainingHud.initialize();
    }

    private static void tick(MinecraftClient client) {
        RLClientRuntime runtime = instance;

        if (runtime == null || client == null) {
            return;
        }

        while (runtime.toggleAiKey.wasPressed()) {
            runtime.aiEnabled = !runtime.aiEnabled;
            runtime.training.setTrainingEnabled(runtime.aiEnabled);

            if (!runtime.aiEnabled) {
                HumanInputBridge.releaseAll(client);
                runtime.training.resetEpisode();
            }
        }

        while (runtime.toggleHudKey.wasPressed()) {
            runtime.hudEnabled = !runtime.hudEnabled;
            TrainingHud.setVisible(runtime.hudEnabled);
        }

        if (!runtime.aiEnabled ||
                client.player == null ||
                client.world == null ||
                client.currentScreen != null) {

            if (!runtime.aiEnabled) {
                HumanInputBridge.releaseAll(client);
            }

            runtime.publishState(false);
            return;
        }

        long tick = client.world.getTime();

        if (tick == runtime.lastTick) {
            return;
        }

        runtime.lastTick = tick;

        PvPEnvironment.Step transition =
                runtime.environment.observe(tick);

        Observation observation =
                runtime.environment.currentObservation(tick);

        if (!observation.isValid()) {
            HumanInputBridge.releaseAll(client);
            runtime.publishState(true);
            return;
        }

        RLTrainingManager.StepResult result =
                runtime.training.step(
                        observation,
                        transition.reward(),
                        transition.done()
                );

        HumanInputBridge.apply(
                client,
                result.action()
        );

        runtime.publishState(true);
    }

    private void publishState(boolean active) {
        var stats = training.lastUpdateStats();

        TrainingHud.setState(
                new TrainingState(
                        aiEnabled,
                        active && training.isTrainingEnabled(),
                        training.environmentSteps(),
                        training.updateCount(),
                        training.episodeReward(),
                        training.meanEpisodeReward(),
                        finite(stats.meanPolicyLoss),
                        finite(stats.meanValueLoss),
                        finite(stats.meanEntropy),
                        finite(stats.meanApproxKl),
                        finite(stats.gradNorm),
                        0.0f
                )
        );
    }

    private static float finite(double value) {
        return Double.isFinite(value) ? (float) value : 0.0f;
    }

    public static boolean isAiEnabled() {
        return instance != null && instance.aiEnabled;
    }

    public static boolean isHudEnabled() {
        return instance == null || instance.hudEnabled;
    }

    public static RLTrainingManager training() {
        if (instance == null) {
            throw new IllegalStateException(
                    "RLClientRuntime has not been initialized"
            );
        }

        return instance.training;
    }
}