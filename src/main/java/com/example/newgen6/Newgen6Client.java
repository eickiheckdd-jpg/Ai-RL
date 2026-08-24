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
    private static KeyBinding toggleCKey;
    private static KeyBinding saveKey;

    @Override
    public void onInitializeClient() {
        controller = new PvpBotController();
        controller.loadWeights();

        // FIX: Replaced String with KeyBinding.Category.MISC
        toggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_P,
                KeyBinding.Category.MISC));
                
        // NEW: AI toggle bound to 'C'
        toggleCKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_c",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                KeyBinding.Category.MISC));

        saveKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.save",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_O,
                KeyBinding.Category.MISC));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            // Checks if either P or C was pressed to toggle the bot
            while (toggleKey.wasPressed() || toggleCKey.wasPressed()) {
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
