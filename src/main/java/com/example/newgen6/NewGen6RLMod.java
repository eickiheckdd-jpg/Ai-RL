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
        AGENT.loadModel("ppo_model.bin");

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

        HudElementRegistry.addLast(Identifier.of(MOD_ID, "hud"), (context, tickCounter) -> {
    MinecraftClient client = MinecraftClient.getInstance();
    if (!showHud || client.options.hudHidden || client.textRenderer == null) return;

    // 1. Draw Translucent Background (ARGB: 0x80 = ~50% Alpha, 0x101010 = Dark Grey)
    int backgroundColor = 0x80101010; 
    context.fill(6, 6, 155, 48, backgroundColor);

    // 2. Draw Borders for better visual clarity
    context.drawBorder(6, 6, 149, 42, 0xFFFFAA00);

    // 3. Render Text Layers with explicit ARGB Hex Colors
    // Title: Gold (0xFFFF00 / ARGB: 0xFFFFAA00)
    context.drawText(client.textRenderer, Text.literal("PPO AI TRAINER"), 12, 10, 0xFFFFAA00, false);

    // AI Status: Green (ON) / Red (OFF)
    String aiStatus = "AI: " + (aiEnabled ? "ON" : "OFF");
    int aiColor = aiEnabled ? 0xFF55FF55 : 0xFFFF5555;
    context.drawText(client.textRenderer, Text.literal(aiStatus), 12, 22, aiColor, false);

    // Reward: Yellow
    String rewardStr = String.format("Reward: %.2f", lastReward);
    context.drawText(client.textRenderer, Text.literal(rewardStr), 12, 34, 0xFFFFFF55, false);
});
