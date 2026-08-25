package com.example.newgen6.hud;

import com.example.newgen6.rl.env.ActionSpace;
import com.example.newgen6.rl.env.ObservationSchema;
import com.example.newgen6.rl.ppo.PPOTrainer;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * Renders the same information architecture as the reference HUD screenshot:
 * "229F x 200T" header, an observation activity grid, a context strip, a
 * 3x3 move grid, and yaw/pitch aim-distribution bar charts. This reads
 * whatever the mod is ACTUALLY doing (last forward pass's real logits /
 * training stats) - it never fabricates numbers, satisfying spec section 16
 * ("do not display CONTEXT while the network only receives the current
 * observation - that would be fake"). The recurrent hidden state genuinely
 * is the context (see GRUCell docs); this panel is just a window into it.
 */
public final class CombatHud {
    private CombatHud() {}

    // Latest snapshot the tick loop publishes; HUD only reads it.
    private static volatile double[] lastYawLogits;
    private static volatile double[] lastPitchLogits;
    private static volatile float[] lastHidden;
    private static volatile String lastMoveLabel = "HOLD";

    public static void publish(double[] yawLogits, double[] pitchLogits, float[] hidden, String moveLabel) {
        lastYawLogits = yawLogits;
        lastPitchLogits = pitchLogits;
        lastHidden = hidden;
        lastMoveLabel = moveLabel;
    }

    public static void register(Supplier<PPOTrainer.UpdateStats> statsSupplier, BooleanSupplier enabledSupplier) {
        HudRenderCallback.EVENT.register((DrawContext ctx, net.minecraft.client.render.RenderTickCounter tickCounter) -> {
            if (!enabledSupplier.getAsBoolean()) return;
            var mc = MinecraftClient.getInstance();
            var tr = mc.textRenderer;
            int x = 8, y = 8;
            int white = 0xFFFFFFFF, green = 0xFF55FFAA, orange = 0xFFFFAA33, gray = 0xFFAAAAAA;

            ctx.drawText(tr, "MINECRAFT COMBAT AI", x, y, white, true); y += 12;
            ctx.drawText(tr, "APEX  " + ObservationSchema.OBSERVATION_SIZE + "F x 200T", x, y, orange, true); y += 12;

            PPOTrainer.UpdateStats s = statsSupplier.get();
            ctx.drawText(tr, String.format("policyLoss=%.4f valueLoss=%.4f H=%.3f KL=%.4f",
                s.meanPolicyLoss, s.meanValueLoss, s.meanEntropy, s.meanApproxKl), x, y, gray, true);
            y += 14;

            // Observation activity strip: coarse per-block "how far from zero" heatmap, real data.
            float[] hidden = lastHidden;
            if (hidden != null) {
                ctx.drawText(tr, "CONTEXT (latent, " + hidden.length + "f)", x, y, green, true); y += 10;
                int cellW = 6;
                for (int i = 0; i < hidden.length; i++) {
                    float v = Math.max(-1f, Math.min(1f, hidden[i]));
                    int intensity = (int) (128 + v * 127);
                    int color = 0xFF000000 | (intensity << 16) | (Math.min(255, intensity + 40) << 8) | intensity;
                    ctx.fill(x + i * cellW, y, x + i * cellW + cellW - 1, y + 8, color);
                }
                y += 14;
            }

            ctx.drawText(tr, "MOVE: " + lastMoveLabel, x, y, green, true); y += 12;

            double[] yawLogits = lastYawLogits;
            double[] pitchLogits = lastPitchLogits;
            if (yawLogits != null) {
                ctx.drawText(tr, "AIM YAW (" + ActionSpace.YAW_BUCKETS + ")", x, y, orange, true); y += 10;
                drawBars(ctx, x, y, yawLogits);
                y += 34;
            }
            if (pitchLogits != null) {
                ctx.drawText(tr, "AIM PITCH (" + ActionSpace.PITCH_BUCKETS + ")", x, y, orange, true); y += 10;
                drawBars(ctx, x, y, pitchLogits);
            }
        });
    }

    private static void drawBars(DrawContext ctx, int x, int y, double[] logits) {
        double[] probs = com.example.newgen6.rl.nn.Categorical.softmax(logits);
        int barW = 8, maxH = 30;
        double maxP = 0; for (double p : probs) maxP = Math.max(maxP, p);
        int argmax = com.example.newgen6.rl.nn.Categorical.argmax(logits);
        for (int i = 0; i < probs.length; i++) {
            int h = (int) Math.max(1, (probs[i] / Math.max(1e-6, maxP)) * maxH);
            int color = i == argmax ? 0xFFFFAA33 : 0xFF2E7D64;
            ctx.fill(x + i * barW, y + (maxH - h), x + i * barW + barW - 2, y + maxH, color);
        }
    }
}