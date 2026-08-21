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

    private static boolean modelInitializationAttempted = false;

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

        /*
         * C toggles the AI.
         */
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
        }

        /*
         * Do absolutely nothing while AI is disabled.
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
         * Initialize ONNX exactly once.
         *
         * This happens after Minecraft is fully running,
         * rather than during Fabric's initial startup.
         */
        if (!modelInitializationAttempted) {

            modelInitializationAttempted = true;

            System.out.println(
                    "NewGen6: Starting ONNX initialization..."
            );

            ModelRunner.initialize();

            if (ModelRunner.isLoaded()) {

                System.out.println(
                        "NewGen6: ONNX initialization successful."
                );

                client.player.sendMessage(
                        Text.literal(
                                "NewGen6: AI model loaded"
                        ),
                        true
                );

            } else {

                System.err.println(
                        "NewGen6: ONNX initialization failed."
                );

                client.player.sendMessage(
                        Text.literal(
                                "NewGen6: AI model failed to load"
                        ),
                        true
                );
            }
        }

        /*
         * IMPORTANT:
         *
         * There is currently NO AI movement,
         * camera control, attacking, jumping, etc.
         *
         * We are only testing that the ONNX model
         * can load successfully.
         */
    }

    public static boolean isEnabled() {
        return enabled;
    }

    public static boolean isModelLoaded() {
        return ModelRunner.isLoaded();
    }
}