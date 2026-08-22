package com.example.newgen6.client;

import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

/**
 * Registers the two client-only keybinds. Call BotKeybinds.register() from
 * your ModInitializer's client entrypoint (Newgen6Client#onInitializeClient).
 */
public class BotKeybinds {

    public static KeyBinding toggleAiState;
    public static KeyBinding toggleHud;

    public static void register() {
        toggleAiState = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai_state",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                "category.newgen6.bot"
        ));

        toggleHud = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                "category.newgen6.bot"
        ));
    }
}
