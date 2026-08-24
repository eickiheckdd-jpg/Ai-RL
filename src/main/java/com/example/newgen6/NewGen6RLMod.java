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
        // Load model and restored STDEV state on launch
        AGENT.loadModel("ppo_model.bin");

        // Safety Shutdown Hook: Auto-saves model state when Minecraft is closed
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[PPO] Auto-saving model on game exit...");
            AGENT.saveModel("ppo_model.bin");
        }));

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
                // 1. Background Box (ARGB: 0x90000000 = Translucent black background)
                context.fill(6, 6, 165, 72, 0x90000000);
                
                // 2. Gold Outline Border
                context.drawBorder(6, 6, 159, 66, 0xFFFFAA00);

                // 3. Status Lines with Explicit ARGB Hex Colors (0xFF forces full opacity)
                context.drawText(client.textRenderer, "PPO AI TRAINER", 12, 10, 0xFFFFAA00, false);

                String aiStatus = "AI: " + (aiEnabled ? "ON" : "OFF");
                int aiColor = aiEnabled ? 0xFF55FF55 : 0xFFFF5555;
                context.drawText(client.textRenderer, aiStatus, 12, 22, aiColor, false);

                context.drawText(client.textRenderer, String.format("Reward: %.2f", lastReward), 12, 34, 0xFFFFFF55, false);
                context.drawText(client.textRenderer, String.format("Noise (STDEV): %.3f", AGENT.stdev), 12, 46, 0xFF55FFFF, false);
                context.drawText(client.textRenderer, "Epsilon: 0.20", 12, 58, 0xFFAAAAFF, false);

            } catch (Exception e) {
                // Prevent HUD rendering exceptions from locking up the UI layer
                e.printStackTrace();
            }
        });
    }
}
