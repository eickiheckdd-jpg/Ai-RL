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

        // Load the ONNX model when the client starts.
        ModelRunner.initialize();

        // C = toggle AI ON/OFF.
        toggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyMapping(
                        "key.newgen6.toggle",
                        InputConstants.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        KeyMapping.Category.MISC
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(NewGen6Client::tick);
    }

    private static void tick(Minecraft client) {

        // Detect C presses.
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

        // Do nothing while disabled.
        if (!enabled) {
            return;
        }

        // Make sure we're actually in a world.
        if (client.player == null || client.level == null) {
            return;
        }

        // Model inference will be connected here next.
    }

    public static boolean isEnabled() {
        return enabled;
    }
}