package com.example.newgen6;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import org.lwjgl.glfw.GLFW;

public final class NewGen6Client implements ClientModInitializer {

    private static KeyMapping toggleKey;
    private static boolean enabled = false;

    @Override
    public void onInitializeClient() {

        // Load and inspect the ONNX model.
        ModelRunner.initialize();

        // C = ON/OFF
        toggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyMapping(
                        "key.newgen6.toggle",
                        InputConstants.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        "key.category.newgen6.ai"
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(NewGen6Client::tick);
    }

    private static void tick(Minecraft client) {

        // Toggle AI with C.
        while (toggleKey.consumeClick()) {
            enabled = !enabled;

            if (client.player != null) {
                client.player.displayClientMessage(
                        Component.literal(
                                "NewGen6 AI: " +
                                        (enabled ? "ON" : "OFF")
                        ),
                        true
                );
            }
        }

        // AI currently does not control anything yet.
        // This stage only verifies that the ONNX model loads correctly.
        if (!enabled) {
            return;
        }

        if (client.player == null || client.level == null) {
            return;
        }

        // Model inference will be connected here after
        // we verify the ONNX input/output tensors.
    }

    public static boolean isEnabled() {
        return enabled;
    }
}