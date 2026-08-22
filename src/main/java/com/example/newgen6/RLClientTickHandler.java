package com.example.newgen6;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.LivingEntity;
import org.lwjgl.glfw.GLFW;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class RLClientTickHandler implements ClientTickEvents.EndTick {
    private final DoubleDQNAgent agent;
    private final TargetSelector targetSelector = new TargetSelector();
    private final RewardCalculator rewardCalculator = new RewardCalculator();
    private final TrainingHudOverlay hudOverlay;
    
    // Dedicated executor and non-blocking lock to prevent queue stacking
    private final ExecutorService trainingExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean isTraining = new AtomicBoolean(false);
    
    private float[] previousState = null;
    private int previousAction = 0;
    private boolean evaluationMode = false;
    private boolean botEnabled = true;
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
            
            // Log environment step immediately on main thread
            agent.addExperience(previousState, previousAction, reward, currentState, done);
            
            if (!evaluationMode && tickCounter % 5 == 0) {
                // If previous batch is still processing, skip this one to preserve FPS
                if (isTraining.compareAndSet(false, true)) {
                    trainingExecutor.submit(() -> {
                        try {
                            agent.trainBatch();
                        } finally {
                            isTraining.set(false); // Release lock for next batch
                        }
                    });
                }
            }

            // Display actual environment ticks vs. backprop cycles
            String stepStats = "E: " + agent.getGameStepCount() + " | T: " + agent.getTrainingStepCount();
            hudOverlay.updateStats(agent.getEpsilon(), reward, CombatAction.values()[previousAction].name(), agent.getTrainingStepCount(), target != null ? target.getName().getString() : "None");

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
        
        // Save dynamically via executor to prevent main thread hitches
        if (tickCounter % 1000 == 0 && !evaluationMode) {
            trainingExecutor.submit(() -> ModelSerializer.saveAgent(agent, modelFile));
        }
    }

    public void setEvaluationMode(boolean eval) {
        this.evaluationMode = eval;
    }

    public void shutdown() {
        trainingExecutor.shutdown();
    }
}
