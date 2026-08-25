package com.example.newgen6.client;

import com.example.newgen6.NewGen6RLMod;
import com.example.newgen6.hud.RlHudOverlay;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class KeyHandler {

    private static KeyBinding toggleAiKey;
    private static KeyBinding toggleHudKey;

    public static void register() {
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                KeyBinding.Category.MISC
        ));

        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                KeyBinding.Category.MISC
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleAiKey.wasPressed()) {
                boolean newState = !NewGen6RLMod.isEnabled();
                NewGen6RLMod.setEnabled(newState);
                if (client.player != null) {
                    client.player.sendMessage(
                        Text.literal("§7[§bNewGen6§7] §fAI Engine: " + (newState ? "§aENABLED" : "§cDISABLED")),
                        true
                    );
                }
            }

            while (toggleHudKey.wasPressed()) {
                boolean newHudState = !RlHudOverlay.isVisible();
                RlHudOverlay.setVisible(newHudState);
                if (client.player != null) {
                    client.player.sendMessage(
                        Text.literal("§7[§bNewGen6§7] §fHUD Overlay: " + (newHudState ? "§aSHOWN" : "§cHIDDEN")),
                        true
                    );
                }
            }
        });
    }
}
