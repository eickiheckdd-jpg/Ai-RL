package com.example.newgen6.rl.env;

import com.example.newgen6.mixin.PvPMixin;

/**
 * Minecraft-facing environment state bridge.
 *
 * Observation/reward only.
 * This class does not execute actions.
 */
public final class PvPEnvironment {

    private final float damageRewardScale;
    private final float damageTakenPenaltyScale;
    private final float victoryReward;
    private final float defeatPenalty;

    private float previousSelfHealth;
    private float previousTargetHealth;
    private boolean episodeActive;

    public PvPEnvironment() {
        this(1.0f, 1.0f, 10.0f, 10.0f);
    }

    public PvPEnvironment(
            float damageRewardScale,
            float damageTakenPenaltyScale,
            float victoryReward,
            float defeatPenalty) {

        if (!Float.isFinite(damageRewardScale)
                || damageRewardScale < 0.0f) {
            throw new IllegalArgumentException(
                    "damageRewardScale must be finite and >= 0");
        }

        if (!Float.isFinite(damageTakenPenaltyScale)
                || damageTakenPenaltyScale < 0.0f) {
            throw new IllegalArgumentException(
                    "damageTakenPenaltyScale must be finite and >= 0");
        }

        if (!Float.isFinite(victoryReward)) {
            throw new IllegalArgumentException(
                    "victoryReward must be finite");
        }

        if (!Float.isFinite(defeatPenalty)) {
            throw new IllegalArgumentException(
                    "defeatPenalty must be finite");
        }

        this.damageRewardScale = damageRewardScale;
        this.damageTakenPenaltyScale = damageTakenPenaltyScale;
        this.victoryReward = victoryReward;
        this.defeatPenalty = defeatPenalty;
    }

    public Step observe(long tick) {

        PvPSnapshot snapshot =
                PvPStateStore.getLatestSnapshot();

        if (snapshot == null || !snapshot.valid) {
            episodeActive = false;

            return new Step(
                    Observation.empty(tick),
                    0.0f,
                    true
            );
        }

        if (!episodeActive) {
            beginEpisode(snapshot);
        }

        float selfHealth = snapshot.selfHealth;
        float targetHealth = snapshot.targetHealth;

        float targetDamage = Math.max(
                0.0f,
                previousTargetHealth - targetHealth
        );

        float selfDamage = Math.max(
                0.0f,
                previousSelfHealth - selfHealth
        );

        float reward =
                targetDamage * damageRewardScale
                - selfDamage * damageTakenPenaltyScale;

        boolean selfDead =
                selfHealth <= 0.0f;

        boolean targetDead =
                snapshot.targetPresent
                && targetHealth <= 0.0f;

        boolean done =
                selfDead
                || targetDead
                || !snapshot.targetPresent;

        if (targetDead) {
            reward += victoryReward;
        }

        if (selfDead) {
            reward -= defeatPenalty;
        }

        previousSelfHealth = selfHealth;

        if (snapshot.targetPresent) {
            previousTargetHealth = targetHealth;
        }

        if (done) {
            episodeActive = false;
        }

        return new Step(
                currentObservation(tick),
                finite(reward),
                done
        );
    }

    public Observation currentObservation(long tick) {

        PvPSnapshot snapshot =
                PvPStateStore.getLatestSnapshot();

        if (snapshot == null || !snapshot.valid) {
            return Observation.empty(tick);
        }

        return Observation.fromSnapshot(
                snapshot,
                tick
        );
    }

    public void reset() {
        episodeActive = false;
        previousSelfHealth = 0.0f;
        previousTargetHealth = 0.0f;
    }

    private void beginEpisode(
            PvPSnapshot snapshot) {

        previousSelfHealth =
                snapshot.selfHealth;

        previousTargetHealth =
                snapshot.targetHealth;

        episodeActive = true;
    }

    public boolean isEpisodeActive() {
        return episodeActive;
    }

    private static float finite(float value) {
        return Float.isFinite(value)
                ? value
                : 0.0f;
    }

    public record Step(
            Observation observation,
            float reward,
            boolean done
    ) {
    }
}