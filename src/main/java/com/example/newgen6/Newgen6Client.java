package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

public class Newgen6Client implements ClientModInitializer {
    public static boolean agentActive = false;
    private static KeyBinding toggleKey;
    private static KeyBinding trainToggleKey;

    private PPOAgent ppoAgent;
    private double[] lastState;
    private int lastAction;

    @Override
    public void onInitializeClient() {
        ppoAgent = new PPOAgent(11, 5); // 11 state features, 5 discrete actions

        toggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_P,
                KeyBinding.Category.MISC));

        trainToggleKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.train",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_O,
                KeyBinding.Category.MISC));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (toggleKey.wasPressed()) {
                agentActive = !agentActive;
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("PPO Agent Active: " + agentActive), false);
                }
            }

            if (trainToggleKey.wasPressed()) {
                ppoAgent.setTraining(!ppoAgent.isTraining());
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("PPO Training Mode: " + ppoAgent.isTraining()), false);
                }
            }

            if (agentActive && client.player != null && client.world != null) {
                double[] currentState = StateExtractor.extractState(client);
                
                if (ppoAgent.isTraining() && lastState != null) {
                    double reward = RewardCalculator.calculateReward(client);
                    ppoAgent.storeTransition(lastState, lastAction, reward, currentState);
                    ppoAgent.trainStep();
                }

                int action = ppoAgent.selectAction(currentState);
                ActionExecutor.execute(action, client);

                lastState = currentState;
                lastAction = action;
            }
        });
    }
}
