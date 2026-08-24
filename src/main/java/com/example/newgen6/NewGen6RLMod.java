package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {
    public static KeyBinding toggleKey;
    public static boolean aiActive = false;

    @Override
    public void onInitializeClient() {
        // Register 'C' Keybinding
        toggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_ai",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C, // Default key set to 'C'
            "category.newgen6.ai"
        ));
    }

    public static void checkToggle(net.minecraft.client.MinecraftClient client) {
        while (toggleKey.wasPressed()) {
            aiActive = !aiActive;
            if (client.player != null) {
                String status = aiActive ? "§a[NewGen6 AI] ENABLED" : "§c[NewGen6 AI] DISABLED";
                client.player.sendMessage(Text.of(status), true); // Actionbar message
            }
        }
    }
}
