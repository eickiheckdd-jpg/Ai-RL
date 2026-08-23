package com.example.newgen6;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class KeyInputHandler {
    private static KeyBinding toggleAgentKey;
    private static KeyBinding toggleHudKey;

    public static void register() {
        toggleAgentKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_agent",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C,
            "category.newgen6.rl"
        ));

        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_hud",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_X,
            "category.newgen6.rl"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleAgentKey.wasPressed()) {
                RLConfig.agentEnabled = !RLConfig.agentEnabled;
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("RL Agent: " + (RLConfig.agentEnabled ? "§aENABLED" : "§cDISABLED")), true);
                }
            }

            while (toggleHudKey.wasPressed()) {
                RLConfig.showDebugHud = !RLConfig.showDebugHud;
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("RL HUD: " + (RLConfig.showDebugHud ? "§aON" : "§cOFF")), true);
                }
            }
        });
    }
}
