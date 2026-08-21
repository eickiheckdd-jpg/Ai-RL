package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

public final class NewGen6Client implements ClientModInitializer {

    private static final KeyBinding.Category AI_CATEGORY =
            KeyBinding.Category.create(
                    Identifier.of("newgen6", "ai")
            );

    private static KeyBinding toggleKey;

    private static boolean enabled = false;

    @Override
    public void onInitializeClient() {

        System.out.println("================================");
        System.out.println("NewGen6: Client initializing...");
        System.out.println("================================");

        toggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.toggle",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        AI_CATEGORY
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(
                NewGen6Client::tick
        );

        System.out.println("NewGen6: C key registered.");
    }

    private static void tick(MinecraftClient client) {

        while (toggleKey.wasPressed()) {

            enabled = !enabled;

            System.out.println(
                    "NewGen6 AI: " +
                    (enabled ? "ON" : "OFF")
            );

            if (client.player != null) {

                client.player.sendMessage(
                        Text.literal(
                                "NewGen6 AI: " +
                                (enabled ? "ON" : "OFF")
                        ),
                        true
                );
            }

            /*
             * Start ONNX loading ONLY when the AI is
             * switched ON for the first time.
             *
             * initializeAsync() does NOT block the
             * Minecraft client thread.
             */
            if (enabled) {

                if (!ModelRunner.isLoaded()
                        && !ModelRunner.isLoading()) {

                    ModelRunner.initializeAsync();

                    if (client.player != null) {
                        client.player.sendMessage(
                                Text.literal(
                                        "NewGen6: Loading AI model..."
                                ),
                                true
                        );
                    }
                }
            }
        }

        /*
         * AI is disabled.
         * Do nothing else this tick.
         */
        if (!enabled) {
            return;
        }

        /*
         * Wait until Minecraft has a player and world.
         */
        if (client.player == null || client.world == null) {
            return;
        }

        /*
         * The ONNX model is still loading.
         *
         * IMPORTANT:
         * We do NOT wait here.
         * We do NOT call the model from the main thread.
         */
        if (ModelRunner.isLoading()) {
            return;
        }

        /*
         * Model failed to load.
         *
         * Don't repeatedly try to load it every tick.
         */
        if (ModelRunner.hasFailed()) {
            return;
        }

        /*
         * The model is ready.
         *
         * This is where we will add actual AI inference
         * later.
         */
        if (ModelRunner.isLoaded()) {

            // AI inference will be added here.

        }
    }

    public static boolean isEnabled() {
        return enabled;
    }
}