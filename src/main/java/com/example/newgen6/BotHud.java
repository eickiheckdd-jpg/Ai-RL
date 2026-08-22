package com.example.newgen6.client;

import com.example.newgen6.rl.AIState;
import com.example.newgen6.rl.BotAction;
import com.example.newgen6.rl.PPOBrain;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.font.TextRenderer;

/**
 * Simple corner overlay — call BotHud.register() once from your client
 * entrypoint. Visibility is controlled by ClientTickController via the
 * 'X' keybind (see BotKeybinds.toggleHud).
 *
 * NOTE: DrawContext#drawText signature is the modern (1.20+) HUD drawing
 * API — verify param order against your 1.21.11 sources if it doesn't
 * compile cleanly.
 */
public class BotHud {

    public static volatile boolean visible = false;

    public static void register() {
        HudRenderCallback.EVENT.register((drawContext, tickDelta) -> {
            if (!visible) return;
            render(drawContext);
        });
    }

    private static void render(DrawContext context) {
        PPOBrain brain = ClientTickController.getBrain();
        AIState state = ClientTickController.getState();
        BotAction lastAction = ClientTickController.getLastAction();
        TextRenderer textRenderer = net.minecraft.client.MinecraftClient.getInstance().textRenderer;

        int x = 6;
        int y = 6;
        int lineHeight = 10;
        int color = 0xFFFFFF;

        context.drawText(textRenderer, "newgen6 PPO", x, y, 0x55FFFF, true);
        y += lineHeight + 2;
        context.drawText(textRenderer, "State: " + state, x, y, color, true);
        y += lineHeight;
        context.drawText(textRenderer, "Action: " + lastAction, x, y, color, true);
        y += lineHeight;
        context.drawText(textRenderer, "Epoch: " + brain.epoch, x, y, color, true);
        y += lineHeight;
        context.drawText(textRenderer, String.format("Last Reward: %.3f", brain.lastReward), x, y, color, true);
        y += lineHeight;
        context.drawText(textRenderer, String.format("Loss: %.4f", brain.lastLoss), x, y, color, true);
        y += lineHeight;
        context.drawText(textRenderer, "Buffer: " + brain.bufferSize + " / " + brain.bufferTarget, x, y, color, true);
    }
}
