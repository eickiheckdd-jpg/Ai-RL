package com.example.newgen6.hud;

import com.example.newgen6.client.NewGen6Client;
import com.example.newgen6.rl.*;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

/**
 * Combat AI overlay styled after the APEX 229F×200T panel.
 * Dark theme, observation grid, context wave, move pad + % sure, aim histograms.
 */
public final class CombatGui {

    private static final int BG = 0xE0101218;
    private static final int PANEL = 0xFF1A1D27;
    private static final int ACCENT = 0xFFFF3B5C;      // red like title
    private static final int GREEN = 0xFF3DFF9A;
    private static final int GREEN_DIM = 0xFF1A8F55;
    private static final int ORANGE = 0xFFFFB020;
    private static final int TEXT = 0xFFE8E8F0;
    private static final int MUTED = 0xFF6B7280;

    // simple ring buffer for context waveform (reward / value signal)
    private static final float[] wave = new float[200];
    private static int waveIdx = 0;

    // aim histogram buckets
    private static final int YAW_BUCKETS = 19;
    private static final int PITCH_BUCKETS = 17;
    private static final float[] yawHist = new float[YAW_BUCKETS];
    private static final float[] pitchHist = new float[PITCH_BUCKETS];

    public static void register() {
        HudRenderCallback.EVENT.register(CombatGui::render);
    }

    public static void pushWave(float value) {
        wave[waveIdx] = value;
        waveIdx = (waveIdx + 1) % wave.length;
    }

    public static void pushAim(float mouseDx, float mouseDy) {
        int yi = (int) MathHelperClamp((mouseDx + 1f) * 0.5f * (YAW_BUCKETS - 1), 0, YAW_BUCKETS - 1);
        int pi = (int) MathHelperClamp((mouseDy + 1f) * 0.5f * (PITCH_BUCKETS - 1), 0, PITCH_BUCKETS - 1);
        // decay
        for (int i = 0; i < YAW_BUCKETS; i++) yawHist[i] *= 0.92f;
        for (int i = 0; i < PITCH_BUCKETS; i++) pitchHist[i] *= 0.92f;
        yawHist[yi] += 1f;
        pitchHist[pi] += 1f;
    }

    private static void render(DrawContext ctx, float tickDelta) {
        if (!NewGen6Client.isDebugVisible()) return;
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.options.hudHidden) return;

        PPOAgent agent = NewGen6Client.AGENT;
        Curriculum cur = agent.getCurriculum();
        StatsLogger stats = agent.getStats();

        int screenW = mc.getWindow().getScaledWidth();
        int panelW = 280;
        int panelH = 420;
        int left = screenW - panelW - 8;
        int top = 8;

        // background panel
        fill(ctx, left, top, left + panelW, top + panelH, BG);
        // header bar
        fill(ctx, left, top, left + panelW, top + 28, PANEL);
        drawCentered(ctx, mc, "MINECRAFT COMBAT AI", left + panelW / 2, top + 6, ACCENT);
        drawCentered(ctx, mc, "APEX  229F × 200T", left + panelW / 2, top + 16, MUTED);

        int y = top + 34;
        int pad = 8;
        int innerL = left + pad;
        int innerR = left + panelW - pad;

        // ----- OBSERVATION grid (229 floats) -----
        drawLabel(ctx, mc, "OBSERVATION", innerL, y, GREEN);
        y += 10;
        int gridW = panelW - pad * 2;
        int cell = 4;
        int cols = gridW / cell;
        int rows = (229 + cols - 1) / cols;
        float[] lastObs = agent.getLastNormObs();
        for (int i = 0; i < 229; i++) {
            int cx = innerL + (i % cols) * cell;
            int cy = y + (i / cols) * cell;
            float v = lastObs != null && i < lastObs.length ? Math.abs(lastObs[i]) : 0f;
            int color = v > 0.6f ? GREEN : (v > 0.25f ? GREEN_DIM : 0xFF1E2230);
            fill(ctx, cx, cy, cx + cell - 1, cy + cell - 1, color);
        }
        y += rows * cell + 4;
        drawLabel(ctx, mc, "229 FLOATS  ·  ONE TICK", innerL, y, MUTED);
        y += 14;

        // ----- CONTEXT wave (200 ticks) -----
        drawLabel(ctx, mc, "CONTEXT", innerL, y, ORANGE);
        y += 10;
        int waveH = 36;
        fill(ctx, innerL, y, innerR, y + waveH, 0xFF12151C);
        float maxW = 0.001f;
        for (float v : wave) if (Math.abs(v) > maxW) maxW = Math.abs(v);
        for (int i = 0; i < 200; i++) {
            int idx = (waveIdx + i) % 200;
            float v = wave[idx] / maxW;
            int barH = (int) (Math.abs(v) * (waveH - 2));
            int color = v >= 0 ? GREEN : ACCENT;
            int bx = innerL + i * (gridW) / 200;
            fill(ctx, bx, y + waveH - barH - 1, bx + 1, y + waveH - 1, color);
        }
        // playhead
        fill(ctx, innerR - 2, y, innerR, y + waveH, ORANGE);
        y += waveH + 2;
        drawLabel(ctx, mc, "200 TICKS  ·  10.0 SECONDS", innerL, y, MUTED);
        y += 14;

        // ----- TRUNK separator -----
        drawCentered(ctx, mc, "—  TRUNK  —", left + panelW / 2, y, MUTED);
        y += 12;

        // ----- MOVE pad + confidence -----
        drawLabel(ctx, mc, "MOVE", innerL, y, TEXT);
        String sure = String.format("%d%% SURE", (int) (agent.lastConfidence * 100));
        ctx.drawText(mc.textRenderer, sure, innerR - mc.textRenderer.getWidth(sure), y, ORANGE, false);
        y += 12;

        int padSize = 14;
        int gridStartX = left + panelW / 2 - padSize * 2;
        // 3x3 movement visual
        boolean[][] cells = movementCells(agent.lastAction);
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int cx = gridStartX + col * (padSize + 2);
                int cy = y + row * (padSize + 2);
                int colr = cells[row][col] ? GREEN : 0xFF2A2F3A;
                fill(ctx, cx, cy, cx + padSize, cy + padSize, colr);
                if (cells[row][col]) {
                    // highlight border
                    fill(ctx, cx, cy, cx + padSize, cy + 1, 0xFFFFFFFF);
                    fill(ctx, cx, cy + padSize - 1, cx + padSize, cy + padSize, 0xFFFFFFFF);
                }
            }
        }
        y += 3 * (padSize + 2) + 6;

        // action buttons
        drawTag(ctx, mc, "JUMP", innerL, y, agent.lastJump, GREEN);
        drawTag(ctx, mc, "SPRINT", innerL + 50, y, agent.lastSprint, GREEN);
        drawTag(ctx, mc, "ATTACK", innerL + 110, y, agent.lastAttack, GREEN);
        drawTag(ctx, mc, "SNEAK", innerL + 175, y, agent.lastSneak, GREEN);
        y += 18;

        // ----- AIM histograms (mouse deltas) -----
        drawLabel(ctx, mc, "AIM", innerL, y, TEXT);
        y += 10;
        drawLabel(ctx, mc, "YAW", innerL, y, MUTED);
        y += 9;
        drawHistogram(ctx, innerL, y, gridW, 22, yawHist, YAW_BUCKETS);
        y += 26;
        drawLabel(ctx, mc, "PITCH", innerL, y, MUTED);
        y += 9;
        drawHistogram(ctx, innerL, y, gridW, 22, pitchHist, PITCH_BUCKETS);
        y += 26;
        drawLabel(ctx, mc, YAW_BUCKETS + " × " + PITCH_BUCKETS + " BUCKETS", innerL, y, MUTED);
        y += 14;

        // footer status
        String mode = agent.isEmergency() ? "EMERGENCY" :
                (!NewGen6Client.isAiEnabled() ? "AI OFF" :
                        (agent.isTraining() ? "TRAINING" : "EVAL"));
        int modeCol = agent.isEmergency() || !NewGen6Client.isAiEnabled() ? ACCENT :
                (agent.isTraining() ? ORANGE : GREEN);
        ctx.drawText(mc.textRenderer, mode, innerL, y, modeCol, false);
        ctx.drawText(mc.textRenderer, "X=HUD  C=AI", innerR - 70, y, MUTED, false);
    }

    // ---------- helpers ----------
    private static boolean[][] movementCells(int action) {
        boolean[][] g = new boolean[3][3];
        // center is idle-ish
        switch (action) {
            case ActionSpace.FORWARD, ActionSpace.SPRINT_FWD, ActionSpace.ATTACK_FWD,
                 ActionSpace.JUMP_FWD, ActionSpace.CLOSE_IN, ActionSpace.CRIT_ATTEMPT -> g[0][1] = true;
            case ActionSpace.BACK, ActionSpace.ATTACK_BACK, ActionSpace.BAIT_BACK -> g[2][1] = true;
            case ActionSpace.LEFT, ActionSpace.STRAFE_LEFT_ATK -> g[1][0] = true;
            case ActionSpace.RIGHT, ActionSpace.STRAFE_RIGHT_ATK -> g[1][2] = true;
            case ActionSpace.FWD_LEFT -> g[0][0] = true;
            case ActionSpace.FWD_RIGHT -> g[0][2] = true;
            case ActionSpace.BACK_LEFT -> g[2][0] = true;
            case ActionSpace.BACK_RIGHT -> g[2][2] = true;
            case ActionSpace.HOLD_DIST, ActionSpace.IDLE -> g[1][1] = true;
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
        int w = mc.textRenderer.getWidth(text) + 6;
        fill(ctx, x, y, x + w, y + 11, on ? onColor : 0xFF2A2F3A);
        ctx.drawText(mc.textRenderer, text, x + 3, y + 2, on ? 0xFF000000 : MUTED, false);
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

    private static float MathHelperClamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}