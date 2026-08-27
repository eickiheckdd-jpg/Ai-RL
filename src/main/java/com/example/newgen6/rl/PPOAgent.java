package com.example.newgen6.rl;

import java.util.Random;

/**
 * Main pure-Java PPO agent with checkpoints, stats, masking, replay, eval mode.
 */
public final class PPOAgent {

    private final MLP net;
    private final RolloutBuffer buffer;
    private final RunningMeanStd obsNorm;
    private final FeatureExtractor features;
    private final RewardCalculator rewardCalc;
    private final Curriculum curriculum;
    private final CheckpointManager checkpoints;
    private final StatsLogger stats;
    private final ActionMask actionMask;
    private final ReplayRecorder replay;
    private final Random rng = new Random();

    private final float[] obs = new float[Config.OBS_DIM];
    private final float[] normObs = new float[Config.OBS_DIM];
    private final float[] advantages = new float[Config.ROLLOUT_STEPS];
    private final float[] returns = new float[Config.ROLLOUT_STEPS];

    private long globalSteps = 0;
    private boolean training = true;
    private boolean emergencyStop = false;

    // last decision (for HUD)
    public int lastAction = 0;
    public float lastReward = 0f;
    public float lastValue = 0f;
    public float[] lastLogits = new float[Config.ACTION_DIM];
    public String lastRewardBreakdown = "";
    public float lastConfidence = 0f;
    public boolean lastJump, lastSprint, lastAttack, lastSneak;
    private final float[] lastNormObsCopy = new float[Config.OBS_DIM];

    public PPOAgent() {
        net = new MLP(Config.OBS_DIM, Config.HIDDEN1, Config.HIDDEN2, Config.ACTION_DIM);
        buffer = new RolloutBuffer(Config.ROLLOUT_STEPS, Config.OBS_DIM);
        obsNorm = new RunningMeanStd(Config.OBS_DIM);
        features = new FeatureExtractor();
        rewardCalc = new RewardCalculator();
        curriculum = new Curriculum();
        checkpoints = new CheckpointManager();
        stats = new StatsLogger();
        actionMask = new ActionMask();
        replay = new ReplayRecorder();

        // try resume
        checkpoints.loadLatest(this);
    }

    public void resetEpisode() {
        features.reset();
        rewardCalc.reset();
    }

    public ActionSpace.Control act(FeatureExtractor.ObsContext ctx,
                                   boolean didHit, boolean didCrit, boolean didTakeDamage,
                                   float dmgDealt, float dmgTaken) {

        if (emergencyStop) {
            return ActionSpace.decode(ActionSpace.IDLE, 0f, 0f, 1f, 0f);
        }

        features.extract(ctx, obs);
        obsNorm.update(obs);
        obsNorm.normalize(obs, normObs);

        MLP.Output out = net.forward(normObs);
        System.arraycopy(out.logits, 0, lastLogits, 0, lastLogits.length);
        lastValue = out.value;

        // action masking
        boolean[] mask = actionMask.compute(ctx);
        actionMask.apply(out.logits, mask);

        float temp = training ? (1.0f + curriculum.getNoiseScale() * 2.5f) : 0.3f;
        int action = net.sample(out.logits, temp);
        float logp = net.logProb(out.logits, action);
        lastAction = action;
        // softmax confidence of chosen action
        float maxL = Float.NEGATIVE_INFINITY;
        for (float l : out.logits) if (l > maxL) maxL = l;
        float sumExp = 0f;
        for (float l : out.logits) sumExp += Math.exp(l - maxL);
        lastConfidence = (float) (Math.exp(out.logits[action] - maxL) / sumExp);

        float yawD = out.look[0];
        float pitchD = out.look[1];
        if (training) {
            yawD += (rng.nextFloat() - 0.5f) * curriculum.getNoiseScale() * 1.5f;
            pitchD += (rng.nextFloat() - 0.5f) * curriculum.getNoiseScale() * 1.2f;
        }

        ActionSpace.Control ctrl = ActionSpace.decode(
                action, yawD, pitchD,
                curriculum.getLookScale(),
                lastConfidence
        );
        lastJump = ctrl.jump;
        lastSprint = ctrl.sprint;
        lastAttack = ctrl.attack;
        lastSneak = ctrl.sneak;
        System.arraycopy(normObs, 0, lastNormObsCopy, 0, Config.OBS_DIM);

        float reward = rewardCalc.compute(ctx, ctrl, didHit, didCrit, didTakeDamage,
                dmgDealt, dmgTaken, curriculum.getStage());
        lastReward = reward;
        lastRewardBreakdown = String.format("r=%.3f hit=%s crit=%s dmg+%.1f/-%.1f",
                reward, didHit, didCrit, dmgDealt, dmgTaken);
        try {
            com.example.newgen6.hud.CombatGui.pushWave(reward);
            com.example.newgen6.hud.CombatGui.pushAim(yawD, pitchD);
        } catch (Throwable ignored) {}

        stats.onStep(reward, dmgDealt, dmgTaken, didHit, didCrit);

        if (training) {
            buffer.add(normObs, action, reward, out.value, logp, out.look, false);
            globalSteps++;
            curriculum.step(globalSteps);

            if (buffer.size() >= Config.ROLLOUT_STEPS) {
                float lastVal = net.forward(normObs).value;
                buffer.computeGAE(lastVal, Config.GAMMA, Config.GAE_LAMBDA, advantages, returns);
                buffer.clear();

                if (globalSteps % 5000 == 0) {
                    checkpoints.saveLatest(this);
                }
            }
        }

        replay.record(globalSteps, normObs, action, reward, dmgDealt, dmgTaken, didHit, didCrit, lastRewardBreakdown);
        return ctrl;
    }

    public void endEpisode(boolean won, boolean lost) {
        stats.endEpisode(won, lost, curriculum.getStage());
        float score = stats.getWinRate() * 100f + stats.getEpReward();
        checkpoints.saveBest(this, score);
        resetEpisode();
    }

    public void emergencyStop() {
        emergencyStop = true;
        training = false;
        System.out.println("[NEWGEN6] !!! EMERGENCY STOP – AI disabled");
    }

    public void clearEmergency() {
        emergencyStop = false;
    }

    public void restoreState(long steps, int stage, float noise, float look, boolean training) {
        this.globalSteps = steps;
        this.training = training;
    }

    public MLP getNetwork() { return net; }
    public Curriculum getCurriculum() { return curriculum; }
    public StatsLogger getStats() { return stats; }
    public CheckpointManager getCheckpoints() { return checkpoints; }
    public ReplayRecorder getReplay() { return replay; }
    public long getGlobalSteps() { return globalSteps; }
    public boolean isTraining() { return training; }
    public void setTraining(boolean t) { training = t; emergencyStop = false; }
    public boolean isEmergency() { return emergencyStop; }
    public FeatureExtractor getFeatures() { return features; }
    public float[] getLastNormObs() { return lastNormObsCopy; }
}
