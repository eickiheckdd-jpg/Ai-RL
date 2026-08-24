package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {
    // 14 State Dimensions, 256 Hidden Size
    public static final PPOAgent AGENT = new PPOAgent(14, 256); 
    
    // Global Toggle State
    public static boolean aiEnabled = false;
    private static KeyBinding toggleAiKey;

    @Override
    public void onInitializeClient() {
        System.out.println("NewGen6 Hybrid PPO Agent Loaded.");

        // 1. Register the KeyBinding to 'C'
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle", 
            InputUtil.Type.KEYSYM, 
            GLFW.GLFW_KEY_C, 
            "category.newgen6.title"
        ));

        // 2. Listen for key presses to toggle the AI state
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleAiKey.wasPressed()) {
                aiEnabled = !aiEnabled;
                
                if (client.player != null) {
                    // Send an Action Bar message (true) to prevent chat spam
                    String status = aiEnabled ? "§aENABLED" : "§cDISABLED";
                    client.player.sendMessage(Text.literal("§8[§bNewGen6 AI§8] §7State: " + status), true);
                }
            }
        });
    }
}
