package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

import java.nio.file.Path;

public class NewGen6RLMod implements ClientModInitializer {
    public static final PPOAgent AGENT = new PPOAgent(30, 256); 
    public static boolean aiEnabled = false;
    public static boolean allowMovement = false; // Starts in Stage 1 (Aim & Attack Only)

    private static KeyBinding toggleAiKey;
    private static KeyBinding toggleMovementKey;
    private static Path brainPath;

    // Create custom category object for Fabric mappings
    private static final KeyBinding.Category CATEGORY = KeyBinding.Category.create(
        Identifier.of("newgen6", "title")
    );

    @Override
    public void onInitializeClient() {
        brainPath = FabricLoader.getInstance().getConfigDir().resolve("newgen6_brain.bin");
        AGENT.loadBrain(brainPath);

        // Keybind 'C' to toggle overall AI state
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle", 
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C, 
            CATEGORY
        ));

        // Keybind 'V' to toggle Curriculum Stage (Movement On/Off)
        toggleMovementKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_movement", 
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_V, 
            CATEGORY
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            // Process AI Toggle
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

            // Process Curriculum Movement Toggle
            while (toggleMovementKey.wasPressed()) {
                allowMovement = !allowMovement;

                if (client.player != null) {
                    String mode = allowMovement ? "§aFULL MOVEMENT (Stage 2)" : "§cAIM & ATTACK ONLY (Stage 1)";
                    client.player.sendMessage(Text.literal("§8[§bNewGen6 Curriculum§8] §7Mode: " + mode), true);
                }
            }
        });
    }
}
