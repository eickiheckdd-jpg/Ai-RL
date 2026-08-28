package com.example.newgen6.hud;

import com.example.newgen6.rl.AgentController;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

public final class NewGen6Hud {
    public static void render(MinecraftClient client, DrawContext ctx, AgentController controller) {
        if (!controller.isHudEnabled()) return;
        if (client.player == null) return;

        int x = 4;
        int y = 4;
        int line = 10;

        draw(ctx, client, x, y, "NEWGEN6 RL", 0xFFFFFF00);
        y += line;

        draw(ctx, client, x, y, "AI: " + controller.stateName(), 0xFF00FF00);
        y += line;

        draw(ctx, client, x, y, "TRAINING: " +
                (controller.getState() == AgentController.AiState.AI_TRAINING ? "ON" : "OFF"), 0xFF00FF00);
        y += line;

        draw(ctx, client, x, y, "EPISODE: " + controller.getEpisode(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "STEP: " + controller.getStep(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "GLOBAL STEP: " + controller.getGlobalStep(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("REWARD: %.3f", controller.getLastReward()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("AVG REWARD: %.3f", controller.getAverageReward()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "W/L: " + controller.getWins() + "/" + controller.getLosses(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "PPO UPDATES: " + controller.getPpoUpdates(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("POLICY LOSS: %.4f", controller.getLastPolicyLoss()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("VALUE LOSS: %.4f", controller.getLastValueLoss()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("ENTROPY: %.4f", controller.getLastEntropy()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "CHECKPOINT: " + controller.getCheckpointStatus(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, "TARGET: " + controller.getTargetName(), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("DIST: %.2f", controller.getTargetDistance()), 0xFFFFFFFF);
        y += line;

        draw(ctx, client, x, y, String.format("TARGET HP: %.2f", controller.getTargetHealth()), 0xFFFFFFFF);
    }

    private static void draw(DrawContext ctx, MinecraftClient client, int x, int y, String text, int color) {
        ctx.drawText(client.textRenderer, text, x, y, color, true);
    }
}