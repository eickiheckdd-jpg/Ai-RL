package com.example.newgen6.client;

import com.example.newgen6.rl.Observation;
import com.example.newgen6.rl.ContextBuffer;
import com.example.newgen6.rl.PolicyNetwork;
import com.example.newgen6.rl.BrainStorage;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.Entity;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

public class NewGen6Client implements ClientModInitializer {

    private final Observation obs = new Observation();
    private final ContextBuffer context = new ContextBuffer();
    private final PolicyNetwork policy = new PolicyNetwork();

    private KeyBinding toggleAiKey;
    private boolean aiActive = false;

    private final float[] YAW_BUCKET_DEGREES = new float[PolicyNetwork.YAW_BINS];
    private final float[] PITCH_BUCKET_DEGREES = new float[PolicyNetwork.PITCH_BINS];

    @Override
    public void onInitializeClient() {
        for (int i = 0; i < PolicyNetwork.YAW_BINS; i++) {
            YAW_BUCKET_DEGREES[i] = -30.0f + (i * (60.0f / (PolicyNetwork.YAW_BINS - 1)));
        }
        for (int i = 0; i < PolicyNetwork.PITCH_BINS; i++) {
            PITCH_BUCKET_DEGREES[i] = -15.0f + (i * (30.0f / (PolicyNetwork.PITCH_BINS - 1)));
        }

        // 1.21 Fabric KeyBinding API update
        KeyBinding.Category aiCategory = KeyBinding.Category.register(Identifier.of("newgen6", "ai"));
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_C, aiCategory
        ));

        ClientTickEvents.END_CLIENT_TICK.register(this::onEndTick);
    }

    private void onEndTick(MinecraftClient client) {
        while (toggleAiKey.wasPressed()) {
            aiActive = !aiActive;
            if (!aiActive) disableAI(client);
        }

        if (!aiActive || client.player == null || client.world == null) return;

        Entity target = client.world.getPlayers().stream()
                .filter(p -> p != client.player)
                .findFirst().orElse(null);

        if (target == null) return;

        float[] currentTickObs = obs.extract(client.player, target);
        context.push(currentTickObs);
        policy.forward(context.getFlattenedContext());

        int moveBucket = policy.sampleCategorical(policy.moveProbs);
        int yawBucket = policy.sampleCategorical(policy.yawProbs);
        int pitchBucket = policy.sampleCategorical(policy.pitchProbs);

        client.options.forwardKey.setPressed(moveBucket == 0 || moveBucket == 1 || moveBucket == 2);
        client.options.backKey.setPressed(moveBucket == 6 || moveBucket == 7 || moveBucket == 8);
        client.options.leftKey.setPressed(moveBucket == 0 || moveBucket == 3 || moveBucket == 6);
        client.options.rightKey.setPressed(moveBucket == 2 || moveBucket == 5 || moveBucket == 8);

        client.options.jumpKey.setPressed(policy.toggleProbs[0] > 0.5f);
        client.options.sprintKey.setPressed(policy.toggleProbs[1] > 0.5f);
        client.options.attackKey.setPressed(policy.toggleProbs[2] > 0.5f);
        client.options.sneakKey.setPressed(policy.toggleProbs[3] > 0.5f);

        client.player.setYaw(client.player.getYaw() + YAW_BUCKET_DEGREES[yawBucket]);
        client.player.setPitch(client.player.getPitch() + PITCH_BUCKET_DEGREES[pitchBucket]);
    }

    private void disableAI(MinecraftClient client) {
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
        client.options.sprintKey.setPressed(false);
        client.options.attackKey.setPressed(false);
        client.options.sneakKey.setPressed(false);
        context.reset();
        BrainStorage.saveBrain(policy);
    }
}
