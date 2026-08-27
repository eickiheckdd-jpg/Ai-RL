package com.example.newgen6.hud;

import com.example.newgen6.client.NewGen6Client;
import com.example.newgen6.rl.*;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

/**
 * Rich debug HUD – toggle with X.
 */
public final class DebugHud {

    public static void register() {
        HudRenderCallback.EVENT.register(DebugHud::render);
    }

    private static void render(DrawContext ctx, float tickDelta) {
        if (!NewGen6Client.isDebugVisible()) return;

        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.options.hudHidden) return;

        PPOAgent agent = NewGen6Client.AGENT;
        Curriculum cur = agent.getCurriculum();
        StatsLogger stats = agent.getStats();

        int x = 4, y = 4;
        int white = 0xE0FFFFFF, green = 0xFF55FF55, yellow = 0xFFFFFF55;
        int red = 0xFFFF5555, cyan = 0xFF55FFFF;

        ctx.drawText(mc.textRenderer, "NEWGEN6 RL — HT3 Sword", x, y, cyan, true);
        y += 12;

        String aiStatus = agent.isEmergency() ? "EMERGENCY STOP" :
                (NewGen6Client.isAiEnabled() ? (agent.isTraining() ? "TRAINING" : "EVAL") : "AI OFF");
        int statusColor = agent.isEmergency() ? red :
                (NewGen6Client.isAiEnabled() ? (agent.isTraining() ? yellow : cyan) : red);
        ctx.drawText(mc.textRenderer, aiStatus, x, y, statusColor, true);
        y += 12;

        ctx.drawText(mc.textRenderer, "Stage: " + cur.stageName(), x, y, green, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("Steps: %d  Episodes: %d",
                agent.getGlobalSteps(), stats.getTotalEpisodes()), x, y, white, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("Noise: %.3f  LookScale: %.2f",
                cur.getNoiseScale(), cur.getLookScale()), x, y, white, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("Entropy: %.4f", cur.getEntropyCoef()), x, y, white, true);
        y += 12;

        ctx.drawText(mc.textRenderer, String.format("WinRate: %.1f%%  W/L: %d/%d",
                stats.getWinRate() * 100f, stats.getWins(), stats.getLosses()), x, y, green, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("EpReward: %.3f  Steps: %d",
                stats.getEpReward(), stats.getEpSteps()), x, y, white, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("Dmg +%.1f / -%.1f",
                stats.getEpDmgDealt(), stats.getEpDmgTaken()), x, y, white, true);
        y += 12;

        ctx.drawText(mc.textRenderer, "Last Action: " + agent.lastAction, x, y, yellow, true);
        y += 10;
        ctx.drawText(mc.textRenderer, agent.lastRewardBreakdown, x, y, white, true);
        y += 10;
        ctx.drawText(mc.textRenderer, String.format("Value: %.3f", agent.lastValue), x, y, white, true);
        y += 12;

        // Top-3 action probabilities
        float[] logits = agent.lastLogits;
        int[] top = topK(logits, 3);
        ctx.drawText(mc.textRenderer, "Top actions:", x, y, cyan, true);
        y += 10;
        float max = Float.NEGATIVE_INFINITY;
        for (float l : logits) if (l > max) max = l;
        float sum = 0f;
        float[] p = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) Math.exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < 3; i++) {
            int a = top[i];
            float prob = sum > 0 ? p[a] / sum : 0f;
            ctx.drawText(mc.textRenderer, String.format("  #%d act %d  %.1f%%", i + 1, a, prob * 100f),
                    x, y, white, true);
            y += 9;
        }

        y += 6;
        ctx.drawText(mc.textRenderer, "X=Debug  C=AI  F6=Save  F7=Eval  F8=Replay  F9=Stop", x, y, 0xA0AAAAAA, true);
    }

    private static int[] topK(float[] arr, int k) {
        int[] idx = new int[k];
        float[] best = new float[k];
        for (int i = 0; i < k; i++) best[i] = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < k; j++) {
                if (arr[i] > best[j]) {
                    for (int m = k - 1; m > j; m--) {
                        best[m] = best[m - 1];
                        idx[m] = idx[m - 1];
                    }
                    best[j] = arr[i];
                    idx[j] = i;
                    break;
                }
            }
        }
        return idx;
    }
}