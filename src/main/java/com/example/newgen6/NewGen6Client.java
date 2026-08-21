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

        System.out.println("NewGen6: Client initializing...");

        toggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyMapping(
                        "key.newgen6.toggle",
                        InputConstants.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        KeyMapping.Category.MISC
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(
                NewGen6Client::tick
        );

        /*
         * IMPORTANT:
         *
         * We do NOT load ONNX during Minecraft's initial startup.
         * Pressing C first enables the system and attempts to load
         * the model.
         *
         * This prevents an ONNX/native-runtime problem from taking
         * down the entire Minecraft client during startup.
         */
    }

    private static void tick(Minecraft client) {

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

            if (enabled && !ModelRunner.isLoaded()) {
                if (client.player != null) {

                    String status =
                            ModelRunner.isLoaded()
                                    ? "Model loaded"
                                    : "Model failed to load";

                    client.player.displayClientMessage(
                            Component.literal(
                                    "NewGen6: " + status
                            ),
                            true
                    );
                }
            }
        }

        if (!enabled) {
            return;
        }

        if (client.player == null) {
            return;
        }

        if (client.level == null) {
            return;
        }

        /*
         * AI inference will be added here after we verify:
         *
         * 1. ONNX Runtime loads.
         * 2. newgen6_full.onnx loads.
         * 3. We can read its input shape.
         * 4. We know exactly what the JSON feature ordering is.
         * 5. We know exactly what each model output/action represents.
         *
         * We should NOT guess those values.
         */
    }

    public static boolean isEnabled() {
        return enabled;
    }
}