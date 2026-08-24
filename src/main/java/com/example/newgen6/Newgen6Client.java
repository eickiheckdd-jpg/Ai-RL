package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class Newgen6Client implements ClientModInitializer {

    public static PvpBotController controller;
    private static KeyBinding toggleKey;
    private static KeyBinding saveKey;

    @Override
    public void onInitializeClient() {
        controller = new PvpBotController();
        controller.loadWeights();

        toggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_P,
                "category.newgen6"));

        saveKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.save",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_O,
                "category.newgen6"));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleKey.wasPressed()) {
                controller.enabled = !controller.enabled;
                if (client.player != null) {
                    client.player.sendMessage(
                            Text.literal("[newgen6] Bot " + (controller.enabled ? "ENABLED" : "disabled")),
                            false);
                }
            }
            while (saveKey.wasPressed()) {
                controller.saveWeights();
            }
            controller.onTick(client);
        });
    }
}
