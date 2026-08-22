package com.example.newgen6.client;

import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

/**
 * Registers the two client-only keybinds. Call BotKeybinds.register() from
 * your ModInitializer's client entrypoint (Newgen6Client#onInitializeClient).
 *
 * NOTE: as of 1.21.9, Fabric requires keybinding categories to be a
 * KeyBinding.Category object (registered once) rather than a raw
 * translation-key string — this is the source of the CATEGORY compile error.
 */
public class BotKeybinds {

    private static final KeyBinding.Category BOT_CATEGORY =
            KeyBinding.Category.create(Identifier.of("newgen6", "bot"));

    public static KeyBinding toggleAiState;
    public static KeyBinding toggleHud;

    public static void register() {
        toggleAiState = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai_state",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                BOT_CATEGORY
        ));

        toggleHud = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                BOT_CATEGORY
        ));
    }
}