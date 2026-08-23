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

    // Operational State
    private float[] previousState = null;
    private int previousAction = 0;
    private boolean evaluationMode = false;
    private boolean botEnabled = true;
    private int tickCounter = 0;

    // Phase 2: Population Pooling & Metrics
    private final File modelFile;
    private final File bestModelFile;
    private final File snapshotDir;
    private int championVersion = 1;
    private final Random random = new Random();
    private double bestNetDamage = -9999.0;
    private double damageDealtInWindow = 0.0;
    private double damageTakenInWindow = 0.0;
    private int windowTicks = 0;

    // Phase 3: Rolling Temporal Memory Buffer (5-tick window, maintains 16-float contract)
    private static final int MEMORY_WINDOW_SIZE = 5;
    private final LinkedList<float[]> temporalMemoryQueue = new LinkedList<>();

    private boolean cKeyWasPressed = false;
    private boolean xKeyWasPressed = false;

    public RLClientTickHandler(TrainingHudOverlay hudOverlay) {
        this.hudOverlay = hudOverlay;
        
        // Locked strictly to 16 inputs to safeguard .bin compatibility
        this.agent = new DoubleDQNAgent(16, CombatAction.values().length);

        File runDir = MinecraftClient.getInstance().runDirectory;
        this.modelFile = new File(runDir, "pvp_rl_model.bin");
        this.bestModelFile = new File(runDir, "best_pvp_model.bin");
        
        this.snapshotDir = new File(runDir, "champion_snapshots");
        if (!snapshotDir.exists()) {
            snapshotDir.mkdirs();
        }

        // Load champion brain if available, else standard model
        if (bestModelFile.exists()) {
            System.out.println("[Newgen6] 🧠 Loading champion brain: best_pvp_model.bin");
            ModelSerializer.loadAgent(agent, bestModelFile);
        } else if (modelFile.exists()) {
            System.out.println("[Newgen6] 📁 Loading standard model: pvp_rl_model.bin");
            ModelSerializer.loadAgent(agent, modelFile);
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

        // Keybindings: C = Toggle Bot, X = Toggle HUD
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

        // Target & Perception processing
        LivingEntity target = targetSelector.findBestTarget(client);
        float[] rawState = PerceptionSystem.getObservation(client, target);
        
        // Phase 3: Temporal memory smoothing
        float[] currentState = processTemporalMemoryState(rawState);
        float aimAlignment = currentState[8]; 

        if (previousState != null) {
            CombatAction prevAct = CombatAction.values()[previousAction];
            boolean wasInvalid = false;
            boolean wasUnnecessaryJump = false;
            boolean wasValidAttack = false;
            boolean wasCrit = false;

            // Attack Evaluation (Spam prevention & Crit detection)
            if (prevAct == CombatAction.ATTACK_SPAM || prevAct == CombatAction.ATTACK_TIMED_SWEEP || prevAct == CombatAction.ATTACK_SPRINT) {
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

            // Jump Evaluation
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

            // Phase 2: Evaluation Window (5,000 ticks)
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

        resetHumanInputs(client);
        executeHumanizedAction(client, action, target);

        previousState = currentState;
        previousAction = actionIndex;
        tickCounter++;

        if (tickCounter % 1000 == 0 && !evaluationMode) {
            trainingExecutor.submit(() -> ModelSerializer.saveAgent(agent, modelFile));
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

    private void executeHumanizedAction(MinecraftClient client, CombatAction action, LivingEntity target) {
        if (client.player == null) return;

        action.execute(client);

        double distanceToTarget = target != null ? client.player.distanceTo(target) : 999.0;
        if (client.options.forwardKey.isPressed()) {
            if (distanceToTarget > 2.2 && !client.player.isSprinting() && client.player.getHungerManager().getFoodLevel() > 6) {
                client.player.setSprinting(true);
            }
        }

        if (target != null) {
            applyHumanizedAimSmoothing(client, target);
        }
    }

    private void applyHumanizedAimSmoothing(MinecraftClient client, LivingEntity target) {
        if (client.player == null || target == null) return;

        double dx = target.getX() - client.player.getX();
        double dy = (target.getY() + target.getStandingEyeHeight()) - (client.player.getY() + client.player.getEyeHeight(client.player.getPose()));
        double dz = target.getZ() - client.player.getZ();
        double distXZ = Math.sqrt(dx * dx + dz * dz);

        float targetYaw = (float) Math.toDegrees(-Math.atan2(dx, dz));
        float targetPitch = (float) Math.toDegrees(-Math.atan2(dy, distXZ));

        float currentYaw = client.player.getYaw();
        float currentPitch = client.player.getPitch();

        float yawDelta = Math.wrapDegrees(targetYaw - currentYaw);
        float pitchDelta = targetPitch - currentPitch;

        double f = client.options.getMouseSensitivity().getValue() * 0.6 + 0.2;
        double gcd = f * f * f * 8.0 * 0.15;

        float smoothYaw = (float) (currentYaw + Math.round(yawDelta * 0.35f / gcd) * gcd);
        float smoothPitch = (float) (currentPitch + Math.round(pitchDelta * 0.35f / gcd) * gcd);

        client.player.setYaw(smoothYaw);
        client.player.setPitch(Math.max(-90.0f, Math.min(90.0f, smoothPitch)));
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
                    ModelSerializer.saveAgent(agent, bestModelFile);
                    ModelSerializer.saveAgent(agent, snapshotFile);
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
                ModelSerializer.loadAgent(agent, randomSnapshot);
            } else if (bestModelFile.exists()) {
                System.out.println("[Newgen6] 🔄 Restoring main champion: best_pvp_model.bin");
                ModelSerializer.loadAgent(agent, bestModelFile);
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
