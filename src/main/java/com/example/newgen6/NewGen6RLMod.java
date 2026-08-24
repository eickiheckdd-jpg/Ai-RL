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

        // Fabric HUD Registration for 1.21.11
        HudElementRegistry.addLast(Identifier.of(MOD_ID, "hud"), (context, tickCounter) -> {
            MinecraftClient client = MinecraftClient.getInstance();
            if (!showHud || client.options.hudHidden || client.textRenderer == null) return;

            // 1. Render Background Box
            context.fill(6, 6, 155, 48, 0x90000000);

            // 2. Translate Matrix Pose forward to guarantee text draws on TOP of the background layer
            context.getMatrices().push();
            context.getMatrices().translate(0, 0, 100);

            // 3. Draw Text Elements
            context.drawText(client.textRenderer, Text.literal("PPO AI TRAINER"), 10, 10, 0xFFAA00, true);
            
            Text aiStatusText = Text.literal("AI: " + (aiEnabled ? "ON" : "OFF"));
            int aiColor = aiEnabled ? 0x55FF55 : 0xFF5555;
            context.drawText(client.textRenderer, aiStatusText, 10, 22, aiColor, true);

            Text rewardText = Text.literal(String.format("Reward: %.2f", lastReward));
            context.drawText(client.textRenderer, rewardText, 10, 34, 0xFFFF55, true);

            // 4. Pop Matrix Pose back to default
            context.getMatrices().pop();
        });
    }
}