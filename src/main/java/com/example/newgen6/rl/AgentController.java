package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;

import java.nio.file.Path;

public final class AgentController {
    public enum AiState {
        OFF,
        AI_PLAYING,
        AI_TRAINING
    }

    private final ObservationBuilder obsBuilder = new ObservationBuilder();
    private final PpoAgent agent = new PpoAgent();
    private final InputController input = new InputController();

    private AiState state = AiState.OFF;
    private boolean hudEnabled = true;

    private int episode = 1;
    private int step = 0;
    private int noTargetTicks = 0;
    private int respawnCooldown = 0;

    private float totalReward = 0.0f;
    private float lastReward = 0.0f;
    private float averageReward = 0.0f;

    private int wins = 0;
    private int losses = 0;

    private boolean lastAttackAction = false;

    private long globalStep = 0;
    private int ticksSinceCheckpoint = 0;
    private int lastSavedPpoUpdates = 0;
    private volatile boolean checkpointPending = false;
    private volatile String checkpointStatus = "none";

    public AgentController() {
        loadCheckpoint();
    }

    public void toggleAi() {
        if (state == AiState.OFF) {
            state = AiState.AI_TRAINING;
            resetEpisode();
        } else {
            if (state == AiState.AI_TRAINING && globalStep > 0) {
                saveCheckpointAsync();
            }

            state = AiState.OFF;
            input.releaseAll(MinecraftClient.getInstance());
            resetEpisode();
        }
    }

    public void toggleHud() {
        hudEnabled = !hudEnabled;
    }

    public boolean isHudEnabled() {
        return hudEnabled;
    }

    public void onClientTick(MinecraftClient client) {
        if (client.world == null || client.player == null || client.currentScreen != null || client.isPaused()) {
            input.releaseAll(client);
            return;
        }

        if (state == AiState.OFF) {
            input.releaseAll(client);
            return;
        }

        if (respawnCooldown > 0) {
            input.releaseAll(client);
            respawnCooldown--;

            if (respawnCooldown == 0) {
                resetEpisode();
            }
            return;
        }

        obsBuilder.update(client);

        float[] obs = obsBuilder.getObservation();

        boolean deterministic = state == AiState.AI_PLAYING;
        ActionSample action = agent.act(obs, deterministic);

        if (action.attack && !lastAttackAction) {
            obsBuilder.onAttackPressed();
        }
        lastAttackAction = action.attack;

        input.apply(client, action);

        float reward = computeReward(client, action, obs);

        boolean win = obsBuilder.consumeWin();
        boolean dead = client.player.getHealth() <= 0.0f;

        if (win) reward += 3.0f;
        if (dead) reward -= 3.0f;

        step++;
        lastReward = reward;
        totalReward += reward;

        if (obsBuilder.getTarget() == null) {
            noTargetTicks++;
        } else {
            noTargetTicks = 0;
        }

        boolean done = false;

        if (dead) {
            losses++;
            done = true;
            respawnCooldown = 40;
        } else if (win) {
            wins++;
            done = true;
            respawnCooldown = 20;
        } else if (step >= AgentConfig.MAX_EPISODE_TICKS || noTargetTicks > 400) {
            done = true;
            respawnCooldown = 20;
        }

        if (state == AiState.AI_TRAINING) {
            agent.store(obs, action, reward, done);

            globalStep++;
            ticksSinceCheckpoint++;

            boolean timeToSave = ticksSinceCheckpoint >= AgentConfig.CHECKPOINT_TICK_INTERVAL;
            boolean updatesToSave = agent.ppoUpdates - lastSavedPpoUpdates >= AgentConfig.CHECKPOINT_UPDATE_INTERVAL;

            if (!checkpointPending && (timeToSave || updatesToSave)) {
                saveCheckpointAsync();
            }
        }

        if (done) {
            averageReward = averageReward * 0.9f + totalReward * 0.1f;
            episode++;
            input.releaseAll(client);
            resetEpisode();
        }
    }

    private float computeReward(MinecraftClient client, ActionSample action, float[] obs) {
        float reward = 0.0f;

        float damageDealt = obsBuilder.consumeTickDamageDealt();
        float damageTaken = obsBuilder.consumeTickDamageTaken();

        boolean hit = obsBuilder.consumeHit();
        boolean miss = obsBuilder.consumeMiss();

        reward += damageDealt * 1.0f;
        reward -= damageTaken * 1.2f;

        if (hit) reward += 0.15f;
        if (miss) reward -= 0.05f;

        if (obsBuilder.getTarget() != null) {
            float dist = obsBuilder.getCurrentDistance();
            boolean visible = obs[62] > 0.5f;

            if (dist <= 3.2f && visible) {
                reward += 0.005f;
            }
        } else {
            reward -= 0.002f;
        }

        // Penalize wasted attacks, but do not decide attacks here.
        if (action.attack && obs[15] < 0.25f) {
            reward -= 0.01f;
        }

        return reward;
    }

    private void resetEpisode() {
        step = 0;
        totalReward = 0.0f;
        noTargetTicks = 0;
        lastAttackAction = false;
        obsBuilder.resetEpisode();
    }

    public void loadCheckpoint() {
        try {
            Path path = ModelCheckpoint.latestPath();
            if (agent.loadCheckpoint(path)) {
                checkpointStatus = "loaded";
            } else {
                checkpointStatus = "none";
            }
        } catch (Throwable t) {
            checkpointStatus = "error";
        }
    }

    public void saveCheckpointAsync() {
        if (checkpointPending) return;

        checkpointPending = true;
        checkpointStatus = "saving";

        agent.saveCheckpointAsync(ModelCheckpoint.latestPath(), episode, globalStep, () -> {
            checkpointPending = false;
            lastSavedPpoUpdates = agent.ppoUpdates;
            ticksSinceCheckpoint = 0;
            checkpointStatus = agent.checkpointStatus;
        });
    }

    public void saveCheckpointNow() {
        try {
            ModelCheckpoint.save(agent, ModelCheckpoint.latestPath(), episode, globalStep);
            checkpointStatus = "saved";
            lastSavedPpoUpdates = agent.ppoUpdates;
            ticksSinceCheckpoint = 0;
        } catch (Throwable t) {
            checkpointStatus = "error";
        }
    }

    public String stateName() {
        return switch (state) {
            case OFF -> "OFF";
            case AI_PLAYING -> "PLAYING";
            case AI_TRAINING -> "TRAINING";
        };
    }

    public AiState getState() {
        return state;
    }

    public int getEpisode() {
        return episode;
    }

    public int getStep() {
        return step;
    }

    public float getLastReward() {
        return lastReward;
    }

    public float getAverageReward() {
        return averageReward;
    }

    public int getWins() {
        return wins;
    }

    public int getLosses() {
        return losses;
    }

    public int getPpoUpdates() {
        return agent.ppoUpdates;
    }

    public float getLastPolicyLoss() {
        return agent.lastPolicyLoss;
    }

    public float getLastValueLoss() {
        return agent.lastValueLoss;
    }

    public float getLastEntropy() {
        return agent.lastEntropy;
    }

    public long getGlobalStep() {
        return globalStep;
    }

    public String getCheckpointStatus() {
        return checkpointStatus;
    }

    public String getTargetName() {
        var t = obsBuilder.getTarget();
        if (t == null) return "none";
        try {
            return t.getDisplayName().getString();
        } catch (Throwable e) {
            return "entity";
        }
    }

    public float getTargetDistance() {
        return obsBuilder.getCurrentDistance();
    }

    public float getTargetHealth() {
        var t = obsBuilder.getTarget();
        if (t == null) return 0.0f;
        return t.getHealth();
    }
}