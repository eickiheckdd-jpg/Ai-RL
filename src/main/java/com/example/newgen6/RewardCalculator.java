package com.example.newgen6.game;

import net.minecraft.entity.player.PlayerEntity;

public class RewardCalculator {

    private float lastSelfHealth = -1;
    private float lastOpponentHealth = -1;

    private static final double TIME_PENALTY = -0.001; // discourage stalling
    private static final double DEATH_PENALTY = -5.0;
    private static final double KILL_REWARD = 5.0;

    public void reset(PlayerEntity self, PlayerEntity opponent) {
        lastSelfHealth = self != null ? self.getHealth() : 20f;
        lastOpponentHealth = opponent != null ? opponent.getHealth() : 20f;
    }

    public double computeReward(PlayerEntity self, PlayerEntity opponent, boolean selfDied, boolean opponentDied) {
        double reward = TIME_PENALTY;

        if (self != null && lastSelfHealth >= 0) {
            double damageTaken = lastSelfHealth - self.getHealth();
            if (damageTaken > 0) reward -= damageTaken;
        }
        if (opponent != null && lastOpponentHealth >= 0) {
            double damageDealt = lastOpponentHealth - opponent.getHealth();
            if (damageDealt > 0) reward += damageDealt;
        }
        if (selfDied) reward += DEATH_PENALTY;
        if (opponentDied) reward += KILL_REWARD;

        if (self != null) lastSelfHealth = self.getHealth();
        if (opponent != null) lastOpponentHealth = opponent.getHealth();
        return reward;
    }
}
