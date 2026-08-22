package com.example.newgen6;

import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

public class TrainingHudOverlay implements HudRenderCallback {
    private boolean enabled = true;
    private float currentEpsilon = 1.0f;
    private float lastReward = 0.0f;
    private String currentActionStr = "NONE";
    private int totalSteps = 0;
    private String targetName = "None";

    public void updateStats(float epsilon, float reward, String action, int steps, String target) {
        this.currentEpsilon = epsilon;
        this.lastReward = reward;
        this.currentActionStr = action;
        this.totalSteps = steps;
        this.targetName = target;
    }

    @Override
    public void onHudRender(DrawContext drawContext, float tickDelta) {
        if (!enabled) return;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.options.hudHidden) return;

        int x = 10;
        int y = 10;
        int color = 0xFFFFFFFF;

        drawContext.drawString(client.textRenderer, "=== RL PvP Agent HUD ===", x, y, 0x00FF00, true);
        drawContext.drawString(client.textRenderer, "Target: " + targetName, x, y + 12, color, true);
        drawContext.drawString(client.textRenderer, "Epsilon: " + String.format("%.3f", currentEpsilon), x, y + 24, color, true);
        drawContext.drawString(client.textRenderer, "Last Reward: " + String.format("%.2f", lastReward), x, y + 36, color, true);
        drawContext.drawString(client.textRenderer, "Action: " + currentActionStr, x, y + 48, color, true);
        drawContext.drawString(client.textRenderer, "Steps: " + totalSteps, x, y + 60, color, true);
    }

    public void toggle() {
        this.enabled = !this.enabled;
    }
}
