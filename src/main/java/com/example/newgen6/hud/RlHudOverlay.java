package com.example.newgen6.hud;

import com.example.newgen6.NewGen6RLMod;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.RenderTickCounter;

public class RlHudOverlay implements HudRenderCallback {

    private static boolean visible = true;

    // Telemetry fields updated by PvPMixin
    public static float lastReward = 0.0f;
    public static float lastValue = 0.0f;
    public static float lastLogProb = 0.0f;
    public static String lastActionSummary = "N/A";

    public static boolean isVisible() {
        return visible;
    }

    public static void setVisible(boolean isVisible) {
        visible = isVisible;
    }

    @Override
    public void onHudRender(DrawContext drawContext, RenderTickCounter tickCounter) {
        if (!visible) return;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.options.hudHidden || client.player == null) return;

        TextRenderer textRenderer = client.textRenderer;

        int x = 10;
        int y = 10;
        int width = 160;
        int height = 75;

        // Semi-transparent dark container background
        drawContext.fill(x, y, x + width, y + height, 0x90000000);
        
        // Manual border rendering for 1.21.11 compatibility
        int borderColor = 0xFF333333;
        drawContext.fill(x, y, x + width, y + 1, borderColor); // Top
        drawContext.fill(x, y + height - 1, x + width, y + height, borderColor); // Bottom
        drawContext.fill(x, y, x + 1, y + height, borderColor); // Left
        drawContext.fill(x + width - 1, y, x + width, y + height, borderColor); // Right

        // Header Title
        drawContext.drawText(textRenderer, "§b§lNEWGEN-6 PPO TELEMETRY", x + 6, y + 6, 0xFFFFFFFF, false);
        drawContext.fill(x + 6, y + 16, x + width - 6, y + 17, 0x50FFFFFF);

        // State & Performance Metrics
        boolean aiActive = NewGen6RLMod.isEnabled();
        String statusText = aiActive ? "§aTRAINING (RUNNING)" : "§cPAUSED";
        drawContext.drawText(textRenderer, "Status: " + statusText, x + 6, y + 22, 0xFFFFFFFF, false);

        String rewardColor = lastReward >= 0 ? "§a+" : "§c";
        drawContext.drawText(textRenderer, String.format("Reward: %s%.3f", rewardColor, lastReward), x + 6, y + 33, 0xFFFFFFFF, false);
        drawContext.drawText(textRenderer, String.format("Critic Val: §e%.3f", lastValue), x + 6, y + 44, 0xFFFFFFFF, false);
        drawContext.drawText(textRenderer, String.format("Log Prob: §d%.3f", lastLogProb), x + 6, y + 55, 0xFFFFFFFF, false);
        drawContext.drawText(textRenderer, "Action: §f" + lastActionSummary, x + 6, y + 66, 0xFFFFFFFF, false);
    }
}
