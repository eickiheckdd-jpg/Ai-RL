package com.example.newgen6.rl;

/** One sampled action + log-prob for PPO. */
public final class ActionSample {
    public int move;
    public boolean jump;
    public boolean sprint;
    public boolean attack;
    public boolean sneak;
    public int yawBucket;
    public int pitchBucket;

    public float logProb;
    public float entropy;
    public float value;

    public float yawDeltaDeg;
    public float pitchDeltaDeg;

    public void applyBuckets() {
        yawDeltaDeg = AimBuckets.yawDeltaDeg(yawBucket);
        pitchDeltaDeg = AimBuckets.pitchDeltaDeg(pitchBucket);
    }

    /**
     * Phase-1 clone: keep aim + PPO stats, force movement/combat off.
     * Policy still samples move heads (entropy), but env ignores them.
     */
    public static ActionSample aimOnly(ActionSample src) {
        ActionSample a = new ActionSample();
        a.move = RLConstants.MOVE_HOLD;
        a.jump = false;
        a.sprint = false;
        a.attack = false;
        a.sneak = false;
        a.yawBucket = src.yawBucket;
        a.pitchBucket = src.pitchBucket;
        a.logProb = src.logProb;
        a.entropy = src.entropy;
        a.value = src.value;
        a.applyBuckets();
        return a;
    }
}
