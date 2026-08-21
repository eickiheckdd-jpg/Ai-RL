package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public final class NewGen6Client implements ClientModInitializer {

    private static KeyBinding toggleKey;
    private static boolean enabled = false;

    @Override
    public void onInitializeClient() {

        System.out.println("NewGen6: Initializing client...");

        // Register the toggle key.
        // GLFW.GLFW_KEY_R is used so we don't need the newer
        // KeyMapping.Category API that caused the previous error.
        toggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.toggle",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_R,
                        "category.newgen6"
                )
        );

        ClientTickEvents.END_CLIENT_TICK.register(client -> {

            while (toggleKey.wasPressed()) {
                enabled = !enabled;

                System.out.println(
                        "NewGen6: AI " +
                        (enabled ? "ENABLED" : "DISABLED")
                );
            }

            if (!enabled) {
                return;
            }

            tick(client);
        });

        // Load ONNX model after the client starts.
        ModelRunner.initialize();

        System.out.println("NewGen6: Client initialized.");
    }

    private static void tick(MinecraftClient client) {

        if (client.player == null || client.world == null) {
            return;
        }

        if (!ModelRunner.isLoaded()) {
            return;
        }

        /*
         * AI processing will go here.
         *
         * For example:
         *
         * 1. Read player state.
         * 2. Read nearby entities.
         * 3. Read target information.
         * 4. Convert those values into the ONNX input tensor.
         * 5. Run the model.
         * 6. Read the model's action output.
         * 7. Convert that action into a Minecraft action.
         *
         * We are intentionally not doing any of that yet.
         */
    }
}