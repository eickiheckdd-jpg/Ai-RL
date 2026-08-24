package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
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
    public static int trainingStep = 0;

    // 30 Inputs, 8 Continuous Actions, 2048 Batch Size
    public static final PPOAgent AGENT = new PPOAgent(30, 8, 2048);
    private static KeyBinding toggleHudKey;
    private static KeyBinding toggleAiKey;

    @Override
    public void onInitializeClient() {
        KeyBinding.Category cat = KeyBinding.Category.create(Identifier.of(MOD_ID, "rl"));
        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.hud", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_X, cat));
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.ai", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_C, cat));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleHudKey.wasPressed()) showHud = !showHud;
            while (toggleAiKey.wasPressed()) aiEnabled = !aiEnabled;
        });

        HudRenderCallback.EVENT.register((drawContext, tick) -> {
            if (!showHud || MinecraftClient.getInstance().options.hudHidden) return;
            drawContext.fill(6, 6, 165, 50, 0x90000000);
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, Text.literal("§6PPO AI TRAINER"), 10, 10, 0xFFFFFF, true);
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, Text.literal("AI: " + (aiEnabled ? "§aON" : "§cOFF")), 10, 22, 0xFFFFFF, true);
            drawContext.drawText(MinecraftClient.getInstance().textRenderer, Text.literal(String.format("Reward: §e%.2f", lastReward)), 10, 34, 0xFFFFFF, true);
        });
    }
}
