package com.example.newgen6;

import com.example.newgen6.rl.PPOAgent;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import org.lwjgl.glfw.GLFW;

public class NewGen6RLMod implements ClientModInitializer {
    public static final String MOD_ID = "newgen6";
    public static boolean aiEnabled = false; 
    public static boolean showHud = true;
    public static float lastReward = 0.0f;

    // Fixed: State dim updated from 30 to 12
    public static final PPOAgent AGENT = new PPOAgent(12, 8, 256);
    private static KeyBinding toggleHudKey;
    private static KeyBinding toggleAiKey;
    private static KeyBinding saveModelKey;

    @Override
    public void onInitializeClient() {
        AGENT.loadModel("ppo_model.bin");

        KeyBinding.Category cat = KeyBinding.Category.create(Identifier.of(MOD_ID, "rl"));
        toggleHudKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.hud", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_X, cat));
        toggleAiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.ai", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_C, cat));
        saveModelKey = KeyBindingHelper.registerKeyBinding(new KeyBinding("key.save", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_V, cat));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleHudKey.wasPressed()) showHud = !showHud;
            while (toggleAiKey.wasPressed()) aiEnabled = !aiEnabled;
            while (saveModelKey.wasPressed()) {
                AGENT.saveModel("ppo_model.bin");
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("§a[PPO] Saved to ppo_model.bin!"), false);
                }
            }
        });

        HudElementRegistry.addLast(Identifier.of(MOD_ID, "hud"), (context, tickCounter) -> {
            MinecraftClient client = MinecraftClient.getInstance();
            if (!showHud || client.options.hudHidden || client.textRenderer == null) return;

            context.fill(6, 6, 165, 50, 0x90000000);
            context.drawTextWithShadow(client.textRenderer, Text.literal("PPO AI TRAINER"), 10, 10, 0xFFAA00);
            context.drawTextWithShadow(client.textRenderer, Text.literal("AI: " + (aiEnabled ? "ON" : "OFF")), 10, 22, aiEnabled ? 0x55FF55 : 0xFF5555);
            context.drawTextWithShadow(client.textRenderer, Text.literal(String.format("Reward: %.2f", lastReward)), 10, 34, 0xFFFF55);
        });
    }
}
