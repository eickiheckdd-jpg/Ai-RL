package com.example.newgen6;

import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.RenderTickCounter;

public class RLDebugHudOverlay implements HudRenderCallback {

    @Override
    public void onHudRender(DrawContext drawContext, RenderTickCounter tickCounter) {
        if (!RLConfig.showDebugHud) return;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) return;

        TextRenderer renderer = client.textRenderer;
        RLTelemetryBus.TelemetrySnapshot data = RLTelemetryBus.get();

        int y = 10;
        int color = 0xFFFFFF;

        drawContext.drawText(renderer, "=== PPO Aim Bot ===", 10, y, 0x00FF00, false); y += 10;
        drawContext.drawText(renderer, "Status: " + (RLConfig.agentEnabled ? "ACTIVE" : "SHADOW / OFF"), 10, y, RLConfig.agentEnabled ? 0x00FF00 : 0xFF0000, false); y += 12;

        drawContext.drawText(renderer, String.format("Yaw Delta: %.2f | Pitch Delta: %.2f", data.yawDelta(), data.pitchDelta()), 10, y, color, false); y += 10;
        drawContext.drawText(renderer, String.format("Val Estimate: %.3f", data.valueEstimate()), 10, y, color, false); y += 10;
        drawContext.drawText(renderer, String.format("Last Reward: %.3f | Ep Reward: %.2f", data.lastReward(), data.totalEpisodeReward()), 10, y, color, false); y += 12;

        drawContext.drawText(renderer, "Obs [YawDiff, PitchDiff, Pres]: " + String.format("%.2f, %.2f, %.0f", data.observation()[1], data.observation()[2], data.observation()[0]), 10, y, 0xAAAAAA, false);
    }
}
