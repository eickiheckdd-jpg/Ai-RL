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
        }

        if (!enabled) {
            return;
        }

        if (client.player == null || client.world == null) {
            return;
        }

        /*
         * AI inference will be added here.
         *
         * We are deliberately not loading ONNX yet.
         * First we want the basic Fabric 1.21.11 client
         * to compile and run reliably.
         */
    }

    public static boolean isEnabled() {
        return enabled;
    }
}