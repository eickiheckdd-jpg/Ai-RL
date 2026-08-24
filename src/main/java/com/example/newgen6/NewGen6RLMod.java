package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {
    public static final String MOD_ID = "newgen6";
    public static boolean aiEnabled = false; 
    public static boolean showHud = true;
    public static float lastReward = 0.0f;

    public static final PPOAgent AGENT = new PPOAgent(12, 8, 256);
    private static KeyBinding toggleHudKey;
    private static KeyBinding toggleAiKey;
    private static KeyBinding saveModelKey;

    @Override
    public void onInitializeClient() {
        // Load existing model weights and STDEV state on launch
        AGENT.loadModel("ppo_model.bin");

        // Save progress on exit
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[PPO] Saving model state on game exit...");
            AGENT.saveModel("ppo_model.bin");
        }));

        // Keybindings
        KeyBinding.Category cat = KeyBinding.Category.create(Identifier.of(MOD_ID, "rl"));
        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.hud", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_X, cat));
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.ai", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_C, cat));
        saveModelKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.save", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_V, cat));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleHudKey.wasPressed()) showHud = !showHud;
            while (toggleAiKey.wasPressed()) aiEnabled = !aiEnabled;
            while (saveModelKey.wasPressed()) {
                AGENT.saveModel("ppo_model.bin");
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("§a[PPO] Saved to ppo_model.bin!"), false);
                }
            }
        });

        // 1.21+ HUD Overlay Integration
        HudElementRegistry.addLast(Identifier.of(MOD_ID, "hud"), (context, tickCounter) -> {
            MinecraftClient client = MinecraftClient.getInstance();
            if (!showHud || client.options.hudHidden || client.textRenderer == null) return;

            try {
                int x1 = 6, y1 = 6, x2 = 165, y2 = 72;
                int borderColor = 0xFFFFAA00;
                int bgColor = 0x90000000;

                // 1. Fill Background Box
                context.fill(x1, y1, x2, y2, bgColor);
                
                // 2. Draw 1-pixel Outline Border (Top, Bottom, Left, Right)
                context.fill(x1, y1, x2, y1 + 1, borderColor);       // Top
                context.fill(x1, y2 - 1, x2, y2, borderColor);       // Bottom
                context.fill(x1, y1, x1 + 1, y2, borderColor);       // Left
                context.fill(x2 - 1, y1, x2, y2, borderColor);       // Right

                // 3. Status Lines
                context.drawText(client.textRenderer, "PPO AI TRAINER", 12, 10, 0xFFFFAA00, false);

                String aiStatus = "AI: " + (aiEnabled ? "ON" : "OFF");
                int aiColor = aiEnabled ? 0xFF55FF55 : 0xFFFF5555;
                context.drawText(client.textRenderer, aiStatus, 12, 22, aiColor, false);

                context.drawText(client.textRenderer, String.format("Reward: %.2f", lastReward), 12, 34, 0xFFFFFF55, false);
                context.drawText(client.textRenderer, String.format("Noise (STDEV): %.3f", AGENT.stdev), 12, 46, 0xFF55FFFF, false);
                context.drawText(client.textRenderer, "Epsilon: 0.20", 12, 58, 0xFFAAAAFF, false);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}
