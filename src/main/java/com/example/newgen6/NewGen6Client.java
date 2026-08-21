package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public final class NewGen6Client implements ClientModInitializer {

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
                        KeyBinding.Category.MISC
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
         * AI inference will be added later.
         *
         * For now this section intentionally does nothing.
         * We are testing the basic Fabric client first.
         */
    }

    public static boolean isEnabled() {
        return enabled;
    }
}