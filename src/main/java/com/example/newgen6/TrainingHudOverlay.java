package com.example.newgen6;

import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

public class TrainingHudOverlay implements HudRenderCallback {
    // 0 = Off, 1 = Mode 1 (Basic Telemetry), 2 = Mode 2 (Advanced Telemetry + Actions)
    private int hudMode = 1;
    
    private float currentReward = 0.0f;
    private double bestNetDamage = 0.0;
    private int bufferSize = 0;
    private float[] lastAction = new float[5];

    // Cycles through: 1 -> 2 -> 0 -> 1 ...
    public void toggle() {
        hudMode = (hudMode + 1) % 3;
    }

    public void updateMetrics(float reward, double bestDamage, int memorySize, float[] action) {
        this.currentReward = reward;
        this.bestNetDamage = bestDamage;
        this.bufferSize = memorySize;
        if (action != null && action.length >= 5) {
            System.arraycopy(action, 0, this.lastAction, 0, 5);
        }
    }

    @Override
    public void onHudRender(DrawContext drawContext, float tickDelta) {
        if (hudMode == 0) return; // HUD is OFF

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) return;

        int x = 10;
        int y = 10;
        int backgroundAlpha = 0xAA000000;

        if (hudMode == 1) {
            // --- MODE 1: Basic Telemetry ---
            drawContext.fill(x - 4, y - 4, x + 205, y + 70, backgroundAlpha);

            drawContext.drawText(client.textRenderer, "§6§l[Newgen6 DDPG - Mode 1]", x, y, 0xFFD700, true);
            drawContext.drawText(client.textRenderer, "§eBot Status: §aACTIVE", x, y + 14, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, String.format("§eReplay Memory: §b%d / 10000", bufferSize), x, y + 28, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, String.format("§eLast Reward: §f%.2f", currentReward), x, y + 40, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, String.format("§eBest Net Damage: §a%.1f", bestNetDamage), x, y + 52, 0xFFFFFF, true);

        } else if (hudMode == 2) {
            // --- MODE 2: Advanced Telemetry + Live Actions ---
            drawContext.fill(x - 4, y - 4, x + 215, y + 115, backgroundAlpha);

            drawContext.drawText(client.textRenderer, "§b§l[Newgen6 DDPG - Mode 2 (Advanced)]", x, y, 0x00FFFF, true);
            drawContext.drawText(client.textRenderer, "§eBot Status: §aACTIVE (Continuous)", x, y + 14, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, "§7Toggles: §f[C] Bot | §f[X] HUD Mode", x, y + 26, 0xCCCCCC, true);

            drawContext.drawText(client.textRenderer, String.format("§eReplay Memory: §b%d / 10000", bufferSize), x, y + 42, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, String.format("§eLast Reward: §f%.2f", currentReward), x, y + 54, 0xFFFFFF, true);
            drawContext.drawText(client.textRenderer, String.format("§eBest Net Damage: §a%.1f", bestNetDamage), x, y + 66, 0xFFFFFF, true);

            drawContext.drawText(client.textRenderer, "§dLive Action Outputs:", x, y + 82, 0xFFFFFF, true);
            String actionString = String.format("§7Y:§f%.2f §7P:§f%.2f §7F:§f%.1f §7S:§f%.1f §7Atk:§f%.1f", 
                    lastAction[0], lastAction[1], lastAction[2], lastAction[3], lastAction[4]);
            drawContext.drawText(client.textRenderer, actionString, x, y + 94, 0xFFFFFF, true);
        }
    }
}
