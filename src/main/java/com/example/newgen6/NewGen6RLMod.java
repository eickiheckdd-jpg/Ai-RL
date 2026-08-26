package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {

    private static KeyBinding aiToggleKey;
    private static KeyBinding hudToggleKey;

    private static boolean aiEnabled = false;
    private static boolean hudEnabled = true;

    @Override
    public void onInitializeClient() {
        aiToggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.ai_toggle",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_C,
                        "category.newgen6"
                )
        );

        hudToggleKey = KeyBindingHelper.registerKeyBinding(
                new KeyBinding(
                        "key.newgen6.hud_toggle",
                        InputUtil.Type.KEYSYM,
                        GLFW.GLFW_KEY_X,
                        "category.newgen6"
                )
        );

        System.out.println("[NEWGEN6] NewGen6 v2 initialized.");
        System.out.println("[NEWGEN6] AI: OFF");
        System.out.println("[NEWGEN6] HUD: ON");
    }

    public static boolean isAiEnabled() {
        return aiEnabled;
    }

    public static boolean isHudEnabled() {
        return hudEnabled;
    }
}