package com.example.newgen6;

import com.example.newgen6.rl.AimEnvironment;
import com.example.newgen6.rl.AimNetwork;
import com.example.newgen6.rl.Observation;
import com.example.newgen6.rl.PPOTrainer;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public final class NewGen6RLMod implements ClientModInitializer {
    private static final String MOD_ID = "newgen6";

    private static KeyBinding toggleAiKey;
    private static KeyBinding reservedHudKey;

    private final AimEnvironment environment = new AimEnvironment();
    private final AimNetwork network = new AimNetwork(0x4E455747454E36L);
    private final PPOTrainer trainer = new PPOTrainer(network);

    private boolean aiEnabled;
    private Observation currentObservation = Observation.noTarget();

    @Override
    public void onInitializeClient() {
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                KeyBinding.Category.MISC));

        reservedHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud_reserved",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                KeyBinding.Category.MISC));

        ClientTickEvents.END_CLIENT_TICK.register(this::tick);
        System.out.println("[NEWGEN6] Loaded Java RL vertical slice. AI starts OFF.");
    }

    private void tick(MinecraftClient client) {
        while (toggleAiKey.wasPressed()) {
            aiEnabled = !aiEnabled;
            System.out.println("[NEWGEN6] AI: " + (aiEnabled ? "ON" : "OFF"));
            if (!aiEnabled) {
                currentObservation = Observation.noTarget();
            }
        }

        while (reservedHudKey.wasPressed()) {
            System.out.println("[NEWGEN6] HUD is reserved for a later verified implementation.");
        }

        if (!aiEnabled || client.player == null || client.world == null) {
            return;
        }

        Observation before = environment.observe(client);
        if (!before.targetValid()) {
            currentObservation = before;
            return;
        }

        AimNetwork.Sample sample = network.sample(before.features());
        environment.applyAction(client.player, sample.yawAction(), sample.pitchAction());

        Observation after = environment.observe(client);
        float reward = environment.reward(before, after);
        boolean done = !after.targetValid();
        double bootstrapValue = done ? 0.0 : network.value(after.features());

        boolean shouldUpdate = trainer.addTransition(
                before.features(),
                sample.rawYaw(),
                sample.rawPitch(),
                sample.yawAction(),
                sample.pitchAction(),
                sample.logProbability(),
                sample.value(),
                reward,
                done,
                before.yawErrorDegrees(),
                before.pitchErrorDegrees());

        currentObservation = after;

        if (shouldUpdate) {
            trainer.update(bootstrapValue);
            System.out.printf(
                    "[NEWGEN6] PPO update=%d steps=%d reward=%.6f meanYawError=%.3f meanPitchError=%.3f policy=%.6f value=%.6f entropy=%.6f paramDelta=%.9f%n",
                    trainer.ppoUpdates(),
                    trainer.totalSteps(),
                    trainer.lastMeanReward(),
                    trainer.lastMeanYawError(),
                    trainer.lastMeanPitchError(),
                    trainer.lastPolicyLoss(),
                    trainer.lastValueLoss(),
                    trainer.lastEntropy(),
                    trainer.lastParameterDelta());
        }
    }

    public boolean isAiEnabled() {
        return aiEnabled;
    }

    public Observation currentObservation() {
        return currentObservation;
    }
}