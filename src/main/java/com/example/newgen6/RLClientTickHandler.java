package com.example.newgen6;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.LivingEntity;
import org.lwjgl.glfw.GLFW;
import java.io.File;

public class RLClientTickHandler implements ClientTickEvents.EndTick {
    private final DoubleDQNAgent agent;
    private final TargetSelector targetSelector = new TargetSelector();
    private final RewardCalculator rewardCalculator = new RewardCalculator();
    private final TrainingHudOverlay hudOverlay;
    
    private float[] previousState = null;
    private int previousAction = 0;
    private boolean evaluationMode = false;
    private boolean botEnabled = true; // Controlled by 'C' key
    private int tickCounter = 0;
    private final File modelFile;
    
    private boolean cKeyWasPressed = false;
    private boolean xKeyWasPressed = false;

    public RLClientTickHandler(TrainingHudOverlay hudOverlay) {
        this.hudOverlay = hudOverlay;
        this.agent = new DoubleDQNAgent(16, CombatAction.values().length);
        this.modelFile = new File(MinecraftClient.getInstance().runDirectory, "pvp_rl_model.bin");
        
        ModelSerializer.loadAgent(agent, modelFile);
    }

    @Override
    public void onEndTick(MinecraftClient client) {
        if (client.player == null || client.world == null || client.isPaused()) return;

        // Pass client.getWindow() directly for 1.21.11 compatibility
        boolean cPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_C);
        if (cPressed && !cKeyWasPressed) {
            botEnabled = !botEnabled;
            System.out.println("[Newgen6] Bot Active: " + botEnabled);
        }
        cKeyWasPressed = cPressed;

        boolean xPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_X);
        if (xPressed && !xKeyWasPressed) {
            hudOverlay.toggle();
        }
        xKeyWasPressed = xPressed;

        if (!botEnabled) return;

        LivingEntity target = targetSelector.findBestTarget(client);
        float[] currentState = PerceptionSystem.getObservation(client, target);
        float aimAlignment = currentState[8]; 

        if (previousState != null) {
            float reward = rewardCalculator.calculateReward(client, target, false, aimAlignment);
            boolean done = client.player.isDead() || (target != null && target.isDead());
            
            if (!evaluationMode) {
                // Throttle training to every 5 ticks to prevent main-thread freezing
                if (tickCounter % 5 == 0) {
                    agent.step(previousState, previousAction, reward, currentState, done);
                }
            }

            hudOverlay.updateStats(agent.getEpsilon(), reward, CombatAction.values()[previousAction].name(), agent.getStepCount(), target != null ? target.getName().getString() : "None");

            if (done) {
                previousState = null;
                rewardCalculator.reset(client, target);
                return;
            }
        }

        int actionIndex = agent.selectAction(currentState, evaluationMode);
        CombatAction action = CombatAction.values()[actionIndex];
        
        client.options.useKey.setPressed(false);
        action.execute(client);

        previousState = currentState;
        previousAction = actionIndex;

        tickCounter++;
        if (tickCounter % 1000 == 0 && !evaluationMode) {
            ModelSerializer.saveAgent(agent, modelFile);
        }
    }

    public void setEvaluationMode(boolean eval) {
        this.evaluationMode = eval;
    }
}
