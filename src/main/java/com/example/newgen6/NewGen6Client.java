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
    private static boolean lastAttack = false;

    @Override
    public void onInitializeClient() {
        toggleKey = KeyBindingHelper.registerKeyBinding(new KeyMapping(
                "key.newgen6.toggle",
                InputConstants.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                "key.category.newgen6.ai"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(NewGen6Client::tick);
    }

    private static void tick(Minecraft client) {
        while (toggleKey.consumeClick()) {
            enabled = !enabled;
            if (client.player != null) {
                client.player.displayClientMessage(
                    Component.literal("NewGen6 AI: " + (enabled ? "ON" : "OFF")),
                    true
                );
            }
        }

        if (!enabled || client.player == null || client.level == null) return;

        // Safety placeholder: inference is intentionally not enabled until the
        // exact 156-feature construction from combat_move_model.json is wired.
        // This prevents feeding incorrectly ordered data to the trained model.
    }

    public static boolean isEnabled() {
        return enabled;
    }
}