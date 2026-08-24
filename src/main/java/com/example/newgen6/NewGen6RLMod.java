package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

import java.nio.file.Path;

public class NewGen6RLMod implements ClientModInitializer {
    // Upgraded to 21 State Dimensions for tactical combat inputs
    public static final PPOAgent AGENT = new PPOAgent(21, 256); 
    public static boolean aiEnabled = false;
    private static KeyBinding toggleAiKey;
    private static Path brainPath;

    @Override
    public void onInitializeClient() {
        brainPath = FabricLoader.getInstance().getConfigDir().resolve("newgen6_brain.bin");
        AGENT.loadBrain(brainPath);

        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle", 
            InputUtil.Type.KEYSYM, 
            GLFW.GLFW_KEY_C, 
            "category.newgen6.title"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleAiKey.wasPressed()) {
                aiEnabled = !aiEnabled;
                
                if (!aiEnabled) {
                    AGENT.saveBrain(brainPath);
                }

                if (client.player != null) {
                    String status = aiEnabled ? "§aENABLED" : "§cDISABLED (Brain Saved)";
                    client.player.sendMessage(Text.literal("§8[§bNewGen6 AI§8] §7State: " + status), true);
                }
            }
        });
    }
}
