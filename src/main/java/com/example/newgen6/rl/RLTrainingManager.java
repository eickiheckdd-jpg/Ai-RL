package com.example.newgen6.rl;

import com.example.newgen6.rl.env.Observation;
import com.example.newgen6.rl.env.ObservationValidator;
import com.example.newgen6.rl.nn.AdamOptimizer;
import com.example.newgen6.rl.nn.PolicyValueNetwork;
import com.example.newgen6.rl.ppo.PPOTrainer;
import com.example.newgen6.rl.ppo.RolloutBuffer;

/**
 * Coordinates rollout collection and PPO updates.
 *
 * This class is deliberately independent from Minecraft input execution.
 * It receives observations, asks the policy for an action, stores the
 * transition, and performs bounded PPO updates when the rollout is full.
 */
public final class RLTrainingManager {

    private final PolicyValueNetwork network;
    private final AdamOptimizer optimizer;
    private final ActionSampler actionSampler;
    private final PPOTrainer trainer;
    private final RolloutBuffer rollout;

    private float[] hiddenState;

    private long environmentSteps;
    private long updateCount;

    private float episodeReward;
    private float meanEpisodeReward;

    private boolean trainingEnabled;

    private PPOTrainer.UpdateStats lastUpdateStats =
            new PPOTrainer.UpdateStats();

    public RLTrainingManager(java.util.Random random) {
        if (random == null) {
            throw new IllegalArgumentException("random cannot be null");
        }

        RLConstants.validateArchitecture();

        this.network = new PolicyValueNetwork(random);

        this.optimizer = new AdamOptimizer(
                RLConstants.LEARNING_RATE,
                RLConstants.ADAM_BETA1,
                RLConstants.ADAM_BETA2,
                RLConstants.ADAM_EPSILON
        );

        network.registerWith(optimizer);

        this.actionSampler = new ActionSampler(random);
        this.trainer = new PPOTrainer(network, optimizer, random);
        this.rollout = new RolloutBuffer();
        this.hiddenState = new float[RLConstants.GRU_HIDDEN_SIZE];
    }

    /**
     * Collects one environment transition.
     *
     * The returned action is only a policy decision. This class does not
     * press keys, move the camera, attack, or otherwise modify Minecraft.
     */
    public StepResult step(
            Observation observation,
            float reward,
            boolean done) {

        if (observation == null) {
            throw new IllegalArgumentException("observation cannot be null");
        }

        ObservationValidator.validate(observation);

        if (!Float.isFinite(reward)) {
            throw new IllegalArgumentException("reward must be finite");
        }

        float[] hiddenIn = hiddenState.clone();

        PolicyValueNetwork.Output output = network.forward(
                observation.copyValues(),
                hiddenIn,
                null
        );

        ActionSampler.Sample sample = actionSampler.sample(output);

        hiddenState = output.hiddenOut.clone();

        rollout.add(
                observation,
                hiddenIn,
                sample.action(),
                sample.logProb(),
                (float) sample.value(),
                reward,
                done
        );

        environmentSteps++;
        episodeReward += reward;

        if (done) {
            meanEpisodeReward =
                    meanEpisodeReward == 0.0f
                            ? episodeReward
                            : meanEpisodeReward * 0.95f
                            + episodeReward * 0.05f;

            episodeReward = 0.0f;
            resetHiddenState();
        }

        boolean updated = false;

        if (trainingEnabled && rollout.isFull()) {
            update();
            updated = true;
        }

        return new StepResult(
                sample.action(),
                sample.logProb(),
                sample.value(),
                updated
        );
    }

    /**
     * Performs one PPO update using the current rollout.
     */
    public void update() {
        if (rollout.isEmpty()) {
            return;
        }

        float bootstrapValue = 0.0f;

        if (!rollout.lastDone()) {
            PolicyValueNetwork.Output output = network.forward(
                    rollout.obs[rollout.lastIndex()],
                    rollout.hiddenIn[rollout.lastIndex()],
                    null
            );

            float value = (float) output.value;
            bootstrapValue = Float.isFinite(value) ? value : 0.0f;
        }

        lastUpdateStats = trainer.update(
                rollout,
                bootstrapValue
        );

        rollout.clear();
        updateCount++;
    }

    public void setTrainingEnabled(boolean enabled) {
        this.trainingEnabled = enabled;
    }

    public boolean isTrainingEnabled() {
        return trainingEnabled;
    }

    public void resetEpisode() {
        rollout.clear();
        episodeReward = 0.0f;
        resetHiddenState();
    }

    public void resetHiddenState() {
        hiddenState = new float[RLConstants.GRU_HIDDEN_SIZE];
    }

    public long environmentSteps() {
        return environmentSteps;
    }

    public long updateCount() {
        return updateCount;
    }

    public float episodeReward() {
        return episodeReward;
    }

    public float meanEpisodeReward() {
        return meanEpisodeReward;
    }

    public RolloutBuffer rollout() {
        return rollout;
    }

    public PPOTrainer.UpdateStats lastUpdateStats() {
        return lastUpdateStats;
    }

    public PolicyValueNetwork network() {
        return network;
    }

    public AdamOptimizer optimizer() {
        return optimizer;
    }

    public record StepResult(
            Action action,
            double logProbability,
            double value,
            boolean updated
    ) {
    }
}