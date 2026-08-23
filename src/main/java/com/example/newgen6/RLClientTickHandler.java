package com.example.newgen6;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.hit.HitResult;
import org.lwjgl.glfw.GLFW;

import java.io.File;
import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class RLClientTickHandler implements ClientTickEvents.EndTick {
    private final DoubleDQNAgent agent;
    private final TargetSelector targetSelector = new TargetSelector();
    private final RewardCalculator rewardCalculator = new RewardCalculator();
    private final TrainingHudOverlay hudOverlay;

    private final ExecutorService trainingExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean isTraining = new AtomicBoolean(false);

    private float[] previousState = null;
    private int previousAction = 0;
    private boolean evaluationMode = false;
    private boolean botEnabled = true;
    private int tickCounter = 0;

    private final File modelFile;
    private final File bestModelFile;
    private final File snapshotDir;
    private int championVersion = 1;
    private final Random random = new Random();
    private double bestNetDamage = -9999.0;
    private double damageDealtInWindow = 0.0;
    private double damageTakenInWindow = 0.0;
    private int windowTicks = 0;

    private static final int MEMORY_WINDOW_SIZE = 5;
    private final LinkedList<float[]> temporalMemoryQueue = new LinkedList<>();

    private boolean cKeyWasPressed = false;
    private boolean xKeyWasPressed = false;

    public RLClientTickHandler(TrainingHudOverlay hudOverlay) {
        this.hudOverlay = hudOverlay;

        // Neural network size automatically adapts to the new CombatAction enum length (9)
        this.agent = new DoubleDQNAgent(16, CombatAction.values().length);

        File runDir = MinecraftClient.getInstance().runDirectory;
        this.modelFile = new File(runDir, "pvp_rl_model.bin");
        this.bestModelFile = new File(runDir, "best_pvp_model.bin");

        this.snapshotDir = new File(runDir, "champion_snapshots");
        if (!snapshotDir.exists()) snapshotDir.mkdirs();

        if (bestModelFile.exists()) {
            System.out.println("[Newgen6] 🧠 Loading champion brain: best_pvp_model.bin");
            ModelSerializer.loadModel(agent, bestModelFile.getAbsolutePath());
        } else if (modelFile.exists()) {
            System.out.println("[Newgen6] 📁 Loading standard model: pvp_rl_model.bin");
            ModelSerializer.loadModel(agent, modelFile.getAbsolutePath());
        } else {
            System.out.println("[Newgen6] ⚠️ No weights found. Initializing fresh network.");
        }
    }

    public void recordDamageDealt(float damage) {
        this.damageDealtInWindow += damage;
    }

    public void recordDamageTaken(float damage) {
        this.damageTakenInWindow += damage;
    }

    @Override
    public void onEndTick(MinecraftClient client) {
        if (client.player == null || client.world == null || client.isPaused()) return;

        boolean cPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_C);
        if (cPressed && !cKeyWasPressed) {
            botEnabled = !botEnabled;
            System.out.println("[Newgen6] Bot Active: " + botEnabled);
            if (!botEnabled) resetHumanInputs(client);
        }
        cKeyWasPressed = cPressed;

        boolean xPressed = InputUtil.isKeyPressed(client.getWindow(), GLFW.GLFW_KEY_X);
        if (xPressed && !xKeyWasPressed) {
            hudOverlay.toggle();
        }
        xKeyWasPressed = xPressed;

        if (!botEnabled) return;

        LivingEntity target = targetSelector.findBestTarget(client);
        float[] rawState = PerceptionSystem.getObservation(client, target);
        float[] currentState = processTemporalMemoryState(rawState);
        float aimAlignment = currentState[8]; 

        if (previousState != null) {
            CombatAction prevAct = CombatAction.values()[previousAction];
            boolean wasInvalid = false;
            boolean wasUnnecessaryJump = false;
            boolean wasValidAttack = false;
            boolean wasCrit = false;

            if (prevAct == CombatAction.ATTACK) {
                if (client.crosshairTarget == null || client.crosshairTarget.getType() != HitResult.Type.ENTITY) {
                    wasInvalid = true; 
                } else {
                    float cooldown = client.player.getAttackCooldownProgress(0.0f);
                    if (cooldown >= 0.85f) {
                        wasValidAttack = true; 
                        if (!client.player.isOnGround() && client.player.getVelocity().y < 0.0) {
                            wasCrit = true; 
                        }
                    } else {
                        wasInvalid = true; 
                    }
                }
            }

            if (prevAct == CombatAction.JUMP && (target == null || aimAlignment > 0.3f)) {
                wasUnnecessaryJump = true;
            }

            float reward = rewardCalculator.calculateReward(client, target, wasInvalid, wasUnnecessaryJump, wasValidAttack, wasCrit, aimAlignment);
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

            windowTicks++;
            if (windowTicks >= 5000 && !evaluationMode) {
                evaluateAndManageBrain();
            }

            hudOverlay.updateStats(
                agent.getEpsilon(), 
                reward, 
                prevAct.name(), 
                agent.getTrainingStepCount(), 
                target != null ? target.getName().getString() : "None"
            );

            if (done) {
                previousState = null;
                temporalMemoryQueue.clear();
                resetHumanInputs(client);
                rewardCalculator.reset(client, target);
                return;
            }
        }

        int actionIndex = agent.selectAction(currentState, evaluationMode);
        CombatAction action = CombatAction.values()[actionIndex];

        // Wipe inputs from the previous tick so the AI has to continuously "hold" buttons
        resetHumanInputs(client);
        
        // Execute the single chosen action directly
        if (client.player != null) {
            action.execute(client);
        }

        previousState = currentState;
        previousAction = actionIndex;
        tickCounter++;

        if (tickCounter % 1000 == 0 && !evaluationMode) {
            trainingExecutor.submit(() -> ModelSerializer.saveModel(agent, modelFile.getAbsolutePath()));
        }
    }

    private float[] processTemporalMemoryState(float[] rawState) {
        temporalMemoryQueue.addLast(rawState);
        if (temporalMemoryQueue.size() > MEMORY_WINDOW_SIZE) {
            temporalMemoryQueue.removeFirst();
        }

        float[] memoryState = new float[16];
        int sampleCount = temporalMemoryQueue.size();

        for (int i = 0; i < 16; i++) {
            float sum = 0.0f;
            for (float[] sample : temporalMemoryQueue) {
                sum += sample[i];
            }
            memoryState[i] = (rawState[i] * 0.70f) + ((sum / sampleCount) * 0.30f);
        }
        return memoryState;
    }

    private void resetHumanInputs(MinecraftClient client) {
        if (client.options == null) return;
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
        client.options.useKey.setPressed(false);
        client.options.attackKey.setPressed(false);
    }

    private void evaluateAndManageBrain() {
        double netDamage = damageDealtInWindow - damageTakenInWindow;

        if (netDamage > bestNetDamage) {
            bestNetDamage = netDamage;
            File snapshotFile = new File(snapshotDir, "champion_v" + championVersion + ".bin");
            championVersion++;

            trainingExecutor.submit(() -> {
                try {
                    ModelSerializer.saveModel(agent, bestModelFile.getAbsolutePath());
                    ModelSerializer.saveModel(agent, snapshotFile.getAbsolutePath());
                    System.out.println("[Newgen6] ⚔️ New Champion Record: " + bestNetDamage + "! Saved " + snapshotFile.getName());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        } else if (netDamage < bestNetDamage - 5.0) {
            System.out.println("[Newgen6] 📉 Performance dropped. Restoring brain...");
            File[] snapshots = snapshotDir.listFiles((dir, name) -> name.endsWith(".bin"));

            if (snapshots != null && snapshots.length > 0 && random.nextFloat() < 0.30f) {
                File randomSnapshot = snapshots[random.nextInt(snapshots.length)];
                System.out.println("[Newgen6] 🎲 Restoring snapshot for diversity: " + randomSnapshot.getName());
                ModelSerializer.loadModel(agent, randomSnapshot.getAbsolutePath());
            } else if (bestModelFile.exists()) {
                System.out.println("[Newgen6] 🔄 Restoring main champion: best_pvp_model.bin");
                ModelSerializer.loadModel(agent, bestModelFile.getAbsolutePath());
            }
        } else {
            System.out.println("[Newgen6] ⚖️ Performance steady. Net Damage: " + netDamage);
        }

        damageDealtInWindow = 0.0;
        damageTakenInWindow = 0.0;
        windowTicks = 0;
    }

    public void setEvaluationMode(boolean eval) {
        this.evaluationMode = eval;
    }

    public void shutdown() {
        trainingExecutor.shutdown();
    }
}
