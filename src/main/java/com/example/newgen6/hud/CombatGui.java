package com.example.newgen6.hud;

import com.example.newgen6.client.NewGen6Client;
import com.example.newgen6.rl.*;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

/**
 * Compact combat AI overlay — top-left corner.
 */
public final class CombatGui {

    private static final int BG = 0xC0101218;
    private static final int PANEL = 0xFF1A1D27;
    private static final int ACCENT = 0xFFFF3B5C;
    private static final int GREEN = 0xFF3DFF9A;
    private static final int GREEN_DIM = 0xFF1A8F55;
    private static final int ORANGE = 0xFFFFB020;
    private static final int TEXT = 0xFFE8E8F0;
    private static final int MUTED = 0xFF6B7280;

    private static final float[] wave = new float[80];
    private static int waveIdx = 0;

    private static final int YAW_BUCKETS = 11;
    private static final int PITCH_BUCKETS = 9;
    private static final float[] yawHist = new float[YAW_BUCKETS];
    private static final float[] pitchHist = new float[PITCH_BUCKETS];

    public static void register() {
        HudRenderCallback.EVENT.register((drawContext, tickCounter) -> CombatGui.render(drawContext, tickCounter));
    }

    public static void pushWave(float value) {
        wave[waveIdx] = value;
        waveIdx = (waveIdx + 1) % wave.length;
    }

    public static void pushAim(float mouseDx, float mouseDy) {
        int yi = (int) clamp((mouseDx + 1f) * 0.5f * (YAW_BUCKETS - 1), 0, YAW_BUCKETS - 1);
        int pi = (int) clamp((mouseDy + 1f) * 0.5f * (PITCH_BUCKETS - 1), 0, PITCH_BUCKETS - 1);
        for (int i = 0; i < YAW_BUCKETS; i++) yawHist[i] *= 0.90f;
        for (int i = 0; i < PITCH_BUCKETS; i++) pitchHist[i] *= 0.90f;
        yawHist[yi] += 1f;
        pitchHist[pi] += 1f;
    }

    private static void render(DrawContext ctx, Object tickCounter) {
        if (!NewGen6Client.isDebugVisible()) return;
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.options.hudHidden) return;

        PPOAgent agent = NewGen6Client.AGENT;
        Curriculum cur = agent.getCurriculum();
        StatsLogger stats = agent.getStats();

        // TOP-LEFT, compact
        int left = 4;
        int top = 4;
        int panelW = 148;
        int panelH = 210;

        fill(ctx, left, top, left + panelW, top + panelH, BG);
        fill(ctx, left, top, left + panelW, top + 18, PANEL);
        drawCentered(ctx, mc, "COMBAT AI", left + panelW / 2, top + 2, ACCENT);
        drawCentered(ctx, mc, "256F×200T", left + panelW / 2, top + 10, MUTED);

        int y = top + 22;
        int pad = 4;
        int innerL = left + pad;
        int innerR = left + panelW - pad;
        int gridW = panelW - pad * 2;

        // OBSERVATION — tiny grid
        drawLabel(ctx, mc, "OBS", innerL, y, GREEN);
        y += 9;
        int cell = 2;
        int cols = gridW / cell;
        float[] lastObs = agent.getLastNormObs();
        int shown = Math.min(256, cols * 8);
        for (int i = 0; i < shown; i++) {
            int cx = innerL + (i % cols) * cell;
            int cy = y + (i / cols) * cell;
            float v = lastObs != null && i < lastObs.length ? Math.abs(lastObs[i]) : 0f;
            int color = v > 0.6f ? GREEN : (v > 0.25f ? GREEN_DIM : 0xFF1E2230);
            fill(ctx, cx, cy, cx + cell - 1, cy + cell - 1, color);
        }
        y += 8 * cell + 3;

        // CONTEXT wave
        drawLabel(ctx, mc, "CTX", innerL, y, ORANGE);
        y += 9;
        int waveH = 18;
        fill(ctx, innerL, y, innerR, y + waveH, 0xFF12151C);
        float maxW = 0.001f;
        for (float v : wave) if (Math.abs(v) > maxW) maxW = Math.abs(v);
        int n = wave.length;
        for (int i = 0; i < n; i++) {
            int idx = (waveIdx + i) % n;
            float v = wave[idx] / maxW;
            int barH = (int) (Math.abs(v) * (waveH - 2));
            int color = v >= 0 ? GREEN : ACCENT;
            int bx = innerL + i * gridW / n;
            fill(ctx, bx, y + waveH - barH - 1, bx + 1, y + waveH - 1, color);
        }
        y += waveH + 3;

        // MOVE + % SURE
        String sure = (int) (agent.lastConfidence * 100) + "%";
        drawLabel(ctx, mc, "MOVE " + sure, innerL, y, TEXT);
        y += 10;
        int padSize = 9;
        int gridStartX = left + panelW / 2 - padSize * 2;
        boolean[][] cells = movementCells(agent.lastAction);
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int cx = gridStartX + col * (padSize + 1);
                int cy = y + row * (padSize + 1);
                fill(ctx, cx, cy, cx + padSize, cy + padSize, cells[row][col] ? GREEN : 0xFF2A2F3A);
            }
        }
        y += 3 * (padSize + 1) + 3;

        // tags
        drawTag(ctx, mc, "J", innerL, y, agent.lastJump, GREEN);
        drawTag(ctx, mc, "SP", innerL + 18, y, agent.lastSprint, GREEN);
        drawTag(ctx, mc, "ATK", innerL + 40, y, agent.lastAttack, GREEN);
        drawTag(ctx, mc, "SN", innerL + 68, y, agent.lastSneak, GREEN);
        y += 14;

        // AIM mini histograms
        drawLabel(ctx, mc, "AIM Y/P", innerL, y, TEXT);
        y += 9;
        drawHistogram(ctx, innerL, y, gridW, 12, yawHist, YAW_BUCKETS);
        y += 14;
        drawHistogram(ctx, innerL, y, gridW, 12, pitchHist, PITCH_BUCKETS);
        y += 16;

        // status line
        String mode = agent.isEmergency() ? "STOP" :
                (!NewGen6Client.isAiEnabled() ? "OFF" :
                        (agent.isTraining() ? "TRAIN" : "EVAL"));
        int modeCol = agent.isEmergency() || !NewGen6Client.isAiEnabled() ? ACCENT :
                (agent.isTraining() ? ORANGE : GREEN);
        ctx.drawText(mc.textRenderer, mode, innerL, y, modeCol, false);
        ctx.drawText(mc.textRenderer, "C=AI X=HUD", innerL + 40, y, MUTED, false);
        y += 10;
        ctx.drawText(mc.textRenderer, "S" + cur.getStage() + " " + agent.getGlobalSteps(), innerL, y, MUTED, false);
    }

    private static boolean[][] movementCells(int action) {
        boolean[][] g = new boolean[3][3];
        switch (action) {
            case ActionSpace.FORWARD, ActionSpace.SPRINT_FWD, ActionSpace.ATK_FWD,
                 ActionSpace.JUMP_FWD, ActionSpace.ATK_SPRINT -> g[0][1] = true;
            case ActionSpace.BACK, ActionSpace.ATK_BACK -> g[2][1] = true;
            case ActionSpace.LEFT, ActionSpace.ATK_LEFT, ActionSpace.JUMP_LEFT,
                 ActionSpace.SPRINT_LEFT -> g[1][0] = true;
            case ActionSpace.RIGHT, ActionSpace.ATK_RIGHT, ActionSpace.JUMP_RIGHT,
                 ActionSpace.SPRINT_RIGHT -> g[1][2] = true;
            case ActionSpace.FWD_LEFT -> g[0][0] = true;
            case ActionSpace.FWD_RIGHT -> g[0][2] = true;
            case ActionSpace.BACK_LEFT -> g[2][0] = true;
            case ActionSpace.BACK_RIGHT -> g[2][2] = true;
            default -> g[1][1] = true;
        }
        return g;
    }

    private static void drawHistogram(DrawContext ctx, int x, int y, int w, int h, float[] hist, int buckets) {
        fill(ctx, x, y, x + w, y + h, 0xFF12151C);
        float max = 0.001f;
        for (float v : hist) if (v > max) max = v;
        int bw = Math.max(1, w / buckets);
        int peak = 0;
        for (int i = 1; i < buckets; i++) if (hist[i] > hist[peak]) peak = i;
        for (int i = 0; i < buckets; i++) {
            int bh = (int) ((hist[i] / max) * (h - 2));
            int color = (i == peak) ? ORANGE : GREEN_DIM;
            int bx = x + i * bw;
            fill(ctx, bx, y + h - bh - 1, bx + bw - 1, y + h - 1, color);
        }
    }

    private static void drawTag(DrawContext ctx, MinecraftClient mc, String text, int x, int y, boolean on, int onColor) {
        int w = mc.textRenderer.getWidth(text) + 4;
        fill(ctx, x, y, x + w, y + 9, on ? onColor : 0xFF2A2F3A);
        ctx.drawText(mc.textRenderer, text, x + 2, y + 1, on ? 0xFF000000 : MUTED, false);
    }

    private static void drawLabel(DrawContext ctx, MinecraftClient mc, String s, int x, int y, int color) {
        ctx.drawText(mc.textRenderer, s, x, y, color, false);
    }

    private static void drawCentered(DrawContext ctx, MinecraftClient mc, String s, int cx, int y, int color) {
        int w = mc.textRenderer.getWidth(s);
        ctx.drawText(mc.textRenderer, s, cx - w / 2, y, color, false);
    }

    private static void fill(DrawContext ctx, int x1, int y1, int x2, int y2, int color) {
        ctx.fill(x1, y1, x2, y2, color);
    }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}