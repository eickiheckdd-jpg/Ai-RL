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
    private final DDPGAgent agent;
    private final TargetSelector targetSelector = new TargetSelector();
    private final RewardCalculator rewardCalculator = new RewardCalculator();
    private final TrainingHudOverlay hudOverlay;

    private final ExecutorService trainingExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean isTraining = new AtomicBoolean(false);

    private float[] previousState = null;
    private float[] previousAction = null;
    private boolean evaluationMode = false;
    private boolean botEnabled = true;
    private int tickCounter = 0;

    private double totalDamageDealt = 0.0;
    private float lastReward = 0.0f;

    private final File modelFile;
    private boolean cKeyWasPressed = false;
    private boolean xKeyWasPressed = false;

    public RLClientTickHandler(TrainingHudOverlay hudOverlay) {
        this.hudOverlay = hudOverlay;
        this.agent = new DDPGAgent(16);

        File runDir = MinecraftClient.getInstance().runDirectory;
        this.modelFile = new File(runDir, "ddpg_pvp_model.bin");

        if (modelFile.exists()) {
            System.out.println("[Newgen6] Loading continuous DDPG model...");
            ModelSerializer.loadModel(agent, modelFile.getAbsolutePath());
        } else {
            System.out.println("[Newgen6] No continuous weights found. Initializing fresh DDPG network.");
        }
    }

    @Override
    public void onEndTick(MinecraftClient client) {
        if (client.player == null || client.world == null || client.isPaused()) return;

        // Toggle Bot Key (C)
        boolean cPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_C);
        if (cPressed && !cKeyWasPressed) {
            botEnabled = !botEnabled;
            System.out.println("[Newgen6] Continuous Bot Active: " + botEnabled);
            if (!botEnabled) resetHumanInputs(client);
        }
        cKeyWasPressed = cPressed;

        // Toggle HUD Key (X - 3-State Cycle)
        boolean xPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_X);
        if (xPressed && !xKeyWasPressed) {
            hudOverlay.toggle();
        }
        xKeyWasPressed = xPressed;

        if (!botEnabled) return;

        LivingEntity target = targetSelector.findBestTarget(client);
        float[] currentState = PerceptionSystem.getObservation(client, target);

        if (previousState != null && previousAction != null) {
            float reward = rewardCalculator.calculateReward(client, target);
            this.lastReward = reward;
            if (reward > 0) totalDamageDealt += reward * 10.0; 

            boolean done = client.player.isDead() || (target != null && target.isDead());

            agent.addExperience(previousState, previousAction, reward, currentState, done);

            if (!evaluationMode && tickCounter % 5 == 0) {
                if (isTraining.compareAndSet(false, true)) {
                    trainingExecutor.submit(() -> {
                        try {
                            agent.trainBatch();
                        } finally {
                            isTraining.set(false);
                        }
                    });
                }
            }

            if (done) {
                previousState = null;
                previousAction = null;
                agent.resetNoise();
                resetHumanInputs(client);
                rewardCalculator.reset(client, target);
                return;
            }
        }

        // Select continuous action array from Actor Network
        float[] actionVector = agent.selectAction(currentState, evaluationMode);

        // Execute smooth movement, aim, and combat input
        ContinuousCombatController.execute(client, actionVector);

        previousState = currentState;
        previousAction = actionVector;
        tickCounter++;

        // Stream live telemetry directly to HUD
        hudOverlay.updateMetrics(lastReward, totalDamageDealt, agent.getMemorySize(), actionVector);

        // Periodic auto-save every 1000 ticks
        if (tickCounter % 1000 == 0 && !evaluationMode) {
            trainingExecutor.submit(() -> ModelSerializer.saveModel(agent, modelFile.getAbsolutePath()));
        }
    }

    private void resetHumanInputs(MinecraftClient client) {
        if (client.options == null) return;
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
        client.options.attackKey.setPressed(false);
    }

    public void setEvaluationMode(boolean eval) {
        this.evaluationMode = eval;
    }

    public void shutdown() {
        trainingExecutor.shutdown();
    }
}
