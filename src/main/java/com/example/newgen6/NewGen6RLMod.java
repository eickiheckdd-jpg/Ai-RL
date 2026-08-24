package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;

public class NewGen6RLMod implements ClientModInitializer {
    public static final String MOD_ID = "newgen6";

    public static boolean aiEnabled = true;
    public static boolean allowMovement = true;
    public static boolean showHud = true;

    public static float lastReward = 0.0f;
    public static float currentStdPitch = 0.0f;
    public static float currentStdYaw = 0.0f;
    public static int trainingStep = 0;

    public static PPOAgent AGENT = new PPOAgent(30);

    @Override
    public void onInitializeClient() {
        HudRenderCallback.EVENT.register((drawContext, tickCounter) -> {
            if (!showHud) return;
            MinecraftClient client = MinecraftClient.getInstance();
            if (client.options.hudHidden) return;

            int x = 10;
            int y = 10;
            int color = 0xFFFFFF;

            drawContext.fill(x - 4, y - 4, x + 165, y + 62, 0x90000000);

            drawContext.drawText(client.textRenderer, "§6=== PPO AI TRAINER ===", x, y, color, true);
            drawContext.drawText(client.textRenderer, "AI Active: " + (aiEnabled ? "§aON" : "§cOFF"), x, y + 12, color, true);
            drawContext.drawText(client.textRenderer, String.format("Last Reward: §e%.2f", lastReward), x, y + 24, color, true);
            drawContext.drawText(client.textRenderer, String.format("Std (P/Y): §b%.3f, %.3f", currentStdPitch, currentStdYaw), x, y + 36, color, true);
            drawContext.drawText(client.textRenderer, "Training Step: §f" + trainingStep, x, y + 48, color, true);
        });
    }
}
