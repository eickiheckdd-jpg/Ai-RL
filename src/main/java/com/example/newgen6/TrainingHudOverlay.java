package com.example.newgen6;

import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.RenderTickCounter;

public class TrainingHudOverlay implements HudRenderCallback {
    private int hudMode = 1;

    private float currentReward = 0.0f;
    private double bestNetDamage = 0.0;
    private int bufferSize = 0;
    private float[] lastAction = new float[6]; // Updated to 6 nodes

    public void toggle() {
        hudMode = (hudMode + 1) % 3;
    }

    public void updateMetrics(float reward, double bestDamage, int memorySize, float[] action) {
        this.currentReward = reward;
        this.bestNetDamage = bestDamage;
        this.bufferSize = memorySize;
        if (action != null && action.length >= 6) {
            System.arraycopy(action, 0, this.lastAction, 0, 6);
        }
    }

    @Override
    public void onHudRender(DrawContext drawContext, RenderTickCounter tickCounter) {
        if (hudMode == 0) return; // HUD is OFF

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) return;

        int x = 12;
        int y = 12;

        if (hudMode == 1) {
            // --- MODE 1: Basic Telemetry ---
            drawContext.fill(x - 6, y - 6, x + 210, y + 72, 0xC0000000);

            drawContext.drawText(client.textRenderer, "Newgen6 DDPG - Mode 1", x, y, 0xFFFFA500, false);
            drawContext.drawText(client.textRenderer, "Bot Status: ACTIVE", x, y + 14, 0xFF55FF55, false);
            drawContext.drawText(client.textRenderer, "Replay Memory: " + bufferSize + " / 10000", x, y + 28, 0xFFFFFF55, false);
            drawContext.drawText(client.textRenderer, String.format("Last Reward: %.2f", currentReward), x, y + 42, 0xFFFFFFFF, false);
            drawContext.drawText(client.textRenderer, String.format("Best Net Damage: %.1f", bestNetDamage), x, y + 56, 0xFF55FFFF, false);

        } else if (hudMode == 2) {
            // --- MODE 2: Advanced Telemetry + Live Actions ---
            drawContext.fill(x - 6, y - 6, x + 240, y + 120, 0xC0000000); // Slightly widened for 6 actions

            drawContext.drawText(client.textRenderer, "Newgen6 DDPG - Mode 2 (Advanced)", x, y, 0xFF00FFFF, false);
            drawContext.drawText(client.textRenderer, "Bot Status: ACTIVE (Continuous)", x, y + 14, 0xFF55FF55, false);
            drawContext.drawText(client.textRenderer, "Replay Memory: " + bufferSize + " / 10000", x, y + 28, 0xFFFFFF55, false);
            drawContext.drawText(client.textRenderer, String.format("Last Reward: %.2f", currentReward), x, y + 42, 0xFFFFFFFF, false);
            drawContext.drawText(client.textRenderer, String.format("Best Net Damage: %.1f", bestNetDamage), x, y + 56, 0xFF55FFFF, false);

            drawContext.drawText(client.textRenderer, "Live Action Outputs (6 Nodes):", x, y + 74, 0xFFFF55FF, false);
            String actionString = String.format("Y:%.2f P:%.2f F:%.1f S:%.1f A:%.1f J:%.1f", 
                    lastAction[0], lastAction[1], lastAction[2], lastAction[3], lastAction[4], lastAction[5]);
            drawContext.drawText(client.textRenderer, actionString, x, y + 88, 0xFFFFFFFF, false);
        }
    }
}
