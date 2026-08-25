package com.example.newgen6.rl.env;

/**
 * Converts real combat outcomes into the scalar reward r_t that PPO
 * maximizes (via GAE/returns - see PPOMath). Kept as a single small,
 * auditable function so the incentive structure is easy to reason about
 * and retune; nothing here touches game state, only already-observed deltas.
 */
public final class RewardFunction {
    private RewardFunction() {}

    public static float TIME_PENALTY = -0.001f;         // small per-tick cost, discourages stalling
    public static float DAMAGE_DEALT_SCALE = 1.0f;       // reward per point of damage dealt
    public static float DAMAGE_TAKEN_SCALE = -1.0f;      // penalty per point of damage taken
    public static float KILL_BONUS = 10.0f;
    public static float DEATH_PENALTY = -10.0f;
    public static float CENTERING_BONUS = 0.002f;        // tiny shaping term for keeping target centered

    public static float compute(float damageDealtThisTick, float damageTakenThisTick,
                                 boolean killedOpponent, boolean died, boolean targetCentered) {
        float r = TIME_PENALTY;
        r += DAMAGE_DEALT_SCALE * damageDealtThisTick;
        r += DAMAGE_TAKEN_SCALE * damageTakenThisTick;
        if (killedOpponent) r += KILL_BONUS;
        if (died) r += DEATH_PENALTY;
        if (targetCentered) r += CENTERING_BONUS;
        return r;
    }
}
