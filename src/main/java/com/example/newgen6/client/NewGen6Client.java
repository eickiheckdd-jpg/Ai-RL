package com.example.newgen6.client;

import com.example.newgen6.rl.CombatAgent;
import com.example.newgen6.rl.RLConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.client.util.InputUtil;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

/**
 * Client entry: C = AI, X = HUD, V = aim-only curriculum toggle.
 * AI starts OFF. aimOnly starts ON (phase-1 aim training).
 *
 * 1.21.11: KeyBinding category is KeyBinding.Category, not a raw String.
 */
public final class NewGen6Client implements ClientModInitializer {
    public static final CombatAgent AGENT = new CombatAgent();

    private static final KeyBinding.Category CATEGORY =
            KeyBinding.Category.create(Identifier.of("newgen6", "main"));

    private static KeyBinding aiToggle;
    private static KeyBinding hudToggle;
    private static KeyBinding aimOnlyToggle;

    @Override
    public void onInitializeClient() {
        aiToggle = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.ai",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_C,
                CATEGORY
        ));
        hudToggle = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.hud",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_X,
                CATEGORY
        ));
        aimOnlyToggle = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.aimonly",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_V,
                CATEGORY
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (aiToggle.wasPressed()) AGENT.toggleAi();
            while (hudToggle.wasPressed()) AGENT.toggleHud();
            while (aimOnlyToggle.wasPressed()) AGENT.toggleAimOnly();
            AGENT.clientTick(client);
        });

        HudRenderCallback.EVENT.register(NewGen6Client::renderHud);
    }

    private static void renderHud(DrawContext ctx, RenderTickCounter tickCounter) {
        if (!AGENT.hudEnabled) return;
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null) return;

        int x = 4;
        int y = 4;
        int color = 0xFFE0E0E0;
        int dim = 0xFF888888;

        ctx.fill(x - 2, y - 2, x + 170, y + 88, 0xC0101018);
        draw(ctx, mc, "NEWGEN6 RL  229F×200T", x, y, color);
        draw(ctx, mc, "AI: " + (AGENT.aiEnabled ? "ON" : "OFF")
                        + "  TRAIN: " + (AGENT.trainEnabled ? "ON" : "OFF"),
                x, y + 10, AGENT.aiEnabled ? 0xFF7CFF7C : 0xFFFF6666);
        draw(ctx, mc, "MODE: " + (AGENT.aimOnly ? "AIM-ONLY" : "FULL"),
                x, y + 20, AGENT.aimOnly ? 0xFF7CC8FF : 0xFFFFC87C);
        draw(ctx, mc, "STEPS: " + AGENT.envSteps + "  UPD: " + AGENT.trainer.updates(), x, y + 30, dim);
        draw(ctx, mc, String.format("REW: %+.3f  MEAN: %+.3f", AGENT.lastReward, AGENT.meanReward), x, y + 40, dim);
        draw(ctx, mc, String.format("AIM: %.3f  Y%d/P%d", AGENT.lastAimError,
                RLConstants.YAW_BUCKETS, RLConstants.PITCH_BUCKETS), x, y + 50, dim);
        if (AGENT.lastAction != null) {
            draw(ctx, mc, String.format("MV:%d J%d S%d A%d SN%d",
                    AGENT.lastAction.move,
                    AGENT.lastAction.jump ? 1 : 0,
                    AGENT.lastAction.sprint ? 1 : 0,
                    AGENT.lastAction.attack ? 1 : 0,
                    AGENT.lastAction.sneak ? 1 : 0), x, y + 60, dim);
            draw(ctx, mc, String.format("AIM BKT Y:%d P:%d",
                    AGENT.lastAction.yawBucket, AGENT.lastAction.pitchBucket), x, y + 70, dim);
        }
        draw(ctx, mc, "CTX: " + AGENT.context.size() + "/" + RLConstants.CONTEXT_TICKS
                + "  C/X/V", x, y + 80, dim);
    }

    private static void draw(DrawContext ctx, MinecraftClient mc, String s, int x, int y, int color) {
        ctx.drawText(mc.textRenderer, s, x, y, color, true);
    }
}