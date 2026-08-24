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

    try {
        // Safe null check on player/agent context
        float currentReward = lastReward;
        float currentStdev = (AGENT != null) ? AGENT.stdev : 0.0f;

        // 1. Draw Background Box (ARGB: 0x90 = Dark semi-transparent background)
        context.fill(6, 6, 160, 72, 0x90000000);
        
        // 2. Optional Outline Border to confirm HUD bounds
        context.drawBorder(6, 6, 154, 66, 0xFFFFAA00);

        // 3. Render Text Stack (0xFF prefix forces full opacity)
        context.drawText(client.textRenderer, "PPO AI TRAINER", 12, 10, 0xFFFFAA00, false);

        String aiStatus = "AI: " + (aiEnabled ? "ON" : "OFF");
        int aiColor = aiEnabled ? 0xFF55FF55 : 0xFFFF5555;
        context.drawText(client.textRenderer, aiStatus, 12, 22, aiColor, false);

        context.drawText(client.textRenderer, String.format("Reward: %.2f", currentReward), 12, 34, 0xFFFFFF55, false);
        context.drawText(client.textRenderer, String.format("Noise (STDEV): %.3f", currentStdev), 12, 46, 0xFF55FFFF, false);
        context.drawText(client.textRenderer, "Epsilon: 0.20", 12, 58, 0xFFAAAAFF, false);

    } catch (Exception e) {
        // Catch any rendering crashes silently to prevent black-box locking
        e.printStackTrace();
    }
});