package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {
    public static KeyBinding toggleAiKey;
    public static KeyBinding toggleHudKey;

    public static boolean aiActive = false;
    public static boolean hudActive = true;

    // HUD Telemetry Metrics
    public static float lastReward = 0.0f;
    public static float currentNoise = 0.25f;
    public static float[] lastActions = new float[7];

    @Override
    public void onInitializeClient() {
        // Toggle AI Key: C (Uses KeyBinding.Category.MISC for 1.21.11 Yarn)
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_ai",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_C,
            KeyBinding.Category.MISC
        ));

        // Toggle HUD Key: X (Uses KeyBinding.Category.MISC for 1.21.11 Yarn)
        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.newgen6.toggle_hud",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_X,
            KeyBinding.Category.MISC
        ));

        // Register In-Game HUD Overlay
        HudRenderCallback.EVENT.register(this::renderHud);
    }

    public static void checkToggles(MinecraftClient client) {
        while (toggleAiKey.wasPressed()) {
            aiActive = !aiActive;
            if (client.player != null) {
                String status = aiActive ? "§a[NewGen6 AI] ENABLED" : "§c[NewGen6 AI] DISABLED";
                client.player.sendMessage(Text.of(status), true);
            }
        }

        while (toggleHudKey.wasPressed()) {
            hudActive = !hudActive;
            if (client.player != null) {
                String status = hudActive ? "§b[NewGen6 HUD] SHOWN" : "§7[NewGen6 HUD] HIDDEN";
                client.player.sendMessage(Text.of(status), true);
            }
        }
    }

    private void renderHud(DrawContext context, RenderTickCounter tickCounter) {
        if (!hudActive) return;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) return;

        int x = 10;
        int y = 10;
        int padding = 6;
        int boxWidth = 190;
        int boxHeight = 85;

        // Translucent dark box (50% opacity black + border)
        int backgroundColor = 0x80000000; 
        int borderColor = 0xFF555555;     

        context.fill(x - 1, y - 1, x + boxWidth + 1, y + boxHeight + 1, borderColor);
        context.fill(x, y, x + boxWidth, y + boxHeight, backgroundColor);

        int colorTitle = 0xFF55FF55; 
        int colorText = 0xFFFFFFFF;  

        context.drawText(client.textRenderer, "=== NewGen6 PPO Telemetry ===", x + padding, y + 6, colorTitle, true);
        context.drawText(client.textRenderer, "AI State: " + (aiActive ? "§aACTIVE" : "§cOFF"), x + padding, y + 18, colorText, true);
        context.drawText(client.textRenderer, String.format("Reward: %.2f", lastReward), x + padding, y + 28, colorText, true);
        context.drawText(client.textRenderer, String.format("Epsilon (Noise): %.4f", currentNoise), x + padding, y + 38, colorText, true);

        context.drawText(client.textRenderer, String.format("Yaw: %.1f | Pitch: %.1f", lastActions[0], lastActions[1]), x + padding, y + 52, colorText, true);
        context.drawText(client.textRenderer, String.format("Move FB: %.2f | Jump: %s", lastActions[2], lastActions[3] > 0.5f ? "§aYES" : "§cNO"), x + padding, y + 62, colorText, true);
        context.drawText(client.textRenderer, String.format("Attack: %s | Strafe: %.2f", lastActions[4] > 0.2f ? "§aSWING" : "§7IDLE", lastActions[5]), x + padding, y + 72, colorText, true);
    }
}
