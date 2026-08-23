package com.example.newgen6;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

public class KeyInputHandler {
    private static KeyBinding toggleAgentKey;
    private static KeyBinding toggleHudKey;
    
    public static final KeyBinding.Category RL_CATEGORY = KeyBinding.Category.create(Identifier.of("newgen6", "rl"));

    public static void register() {
        toggleAgentKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_agent", GLFW.GLFW_KEY_C, RL_CATEGORY
        ));

        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_hud", GLFW.GLFW_KEY_X, RL_CATEGORY
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
