
package com.example.newgen6.hud;

import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;

/**
 * Small live RL training overlay.
 *
 * X controls visibility. The rendering layer is independent from PPO.
 */
public final class TrainingHud {

    private static final int X = 6;
    private static final int Y = 6;
    private static final int WIDTH = 250;

    private static volatile boolean visible = true;
    private static volatile TrainingState state =
            TrainingState.empty();

    private TrainingHud() {
    }

    public static void initialize() {
        HudRenderCallback.EVENT.register(
                (drawContext, tickCounter) ->
                        render(drawContext)
        );
    }

    public static void setVisible(boolean visibleNow) {
        visible = visibleNow;
    }

    public static boolean isVisible() {
        return visible;
    }

    public static void setState(TrainingState newState) {
        if (newState != null) {
            state = newState;
        }
    }

    public static TrainingState getState() {
        return state;
    }

    private static void render(DrawContext context) {
        if (!visible) {
            return;
        }

        MinecraftClient client =
                MinecraftClient.getInstance();

        if (client == null ||
                client.textRenderer == null ||
                client.player == null) {
            return;
        }

        int rows = state.training() ? 11 : 5;
        int height = 18 + rows * 12;

        context.fill(
                X,
                Y,
                X + WIDTH,
                Y + height,
                0xB8000000
        );

        int y = Y + 5;

        draw(context, "NEWGEN6 RL", y, 0xFFFFFF);
        y += 13;

        draw(
                context,
                "AI  " + (state.aiEnabled() ? "ON" : "OFF")
                        + "    TRAIN  "
                        + (state.training() ? "ON" : "OFF"),
                y,
                state.aiEnabled() ? 0x55FF55 : 0xFF7777
        );
        y += 12;

        draw(
                context,
                "STEPS     " + state.environmentSteps(),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "UPDATES   " + state.updates(),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "EP REWARD " + format(state.episodeReward()),
                y,
                0xE8E8E8
        );
        y += 12;

        if (!state.training()) {
            draw(
                    context,
                    "C: AI   X: HUD",
                    y,
                    0xAAAAAA
            );
            return;
        }

        draw(
                context,
                "MEAN REWARD " + format(state.meanReward()),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "POLICY      " + format(state.policyLoss()),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "VALUE       " + format(state.valueLoss()),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "ENTROPY     " + format(state.entropy()),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "APPROX KL   " + format(state.approxKl()),
                y,
                0xE8E8E8
        );
        y += 12;

        draw(
                context,
                "GRAD NORM   " + format(state.gradientNorm()),
                y,
                0xE8E8E8
        );
    }

    private static void draw(
            DrawContext context,
            String text,
            int y,
            int color) {

        MinecraftClient client =
                MinecraftClient.getInstance();

        context.drawText(
                client.textRenderer,
                text,
                X + 6,
                y,
                color,
                false
        );
    }

    private static String format(float value) {
        return String.format(
                java.util.Locale.ROOT,
                "%.4f",
                value
        );
    }
}