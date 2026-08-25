package com.example.newgen6.rl.env;

/**
 * Defines the factorized discrete action space. The policy has multiple
 * independent categorical heads (NOT one giant combinatorial action),
 * matching the HUD reference (MOVE grid + JUMP/SPRINT/SNEAK/ATTACK toggles
 * + separate 19x17 AIM distribution).
 *
 * Joint action log-probability (see PolicyValueNetwork#logProbJoint):
 *   logP(a) = logP(move) + logP(yaw) + logP(pitch)
 *           + logP(jump) + logP(sprint) + logP(sneak) + logP(attack)
 *
 * All heads are softmax categoricals (including the binary toggles, encoded
 * as 2-way categoricals: index 0 = OFF, index 1 = ON) so a single sampling /
 * log-prob / entropy code path (see Categorical.java) covers every head.
 */
public final class ActionSpace {
    private ActionSpace() {}

    /** 3x3 movement grid from the HUD: 8 directions + neutral/hold. */
    public static final int MOVE_ACTIONS = 9;
    public static final int MOVE_HOLD       = 4; // center cell
    public static final int MOVE_FORWARD    = 1; // top-center
    public static final int MOVE_BACKWARD   = 7; // bottom-center
    public static final int MOVE_LEFT       = 3; // mid-left
    public static final int MOVE_RIGHT      = 5; // mid-right
    public static final int MOVE_FWD_LEFT   = 0;
    public static final int MOVE_FWD_RIGHT  = 2;
    public static final int MOVE_BACK_LEFT  = 6;
    public static final int MOVE_BACK_RIGHT = 8;

    /** Aim ABI: exactly 19 yaw buckets x 17 pitch buckets, per spec. */
    public static final int YAW_BUCKETS = 19;
    public static final int PITCH_BUCKETS = 17;

    /** Binary toggle heads, each a 2-way categorical: [OFF, ON]. */
    public static final int TOGGLE_ACTIONS = 2;

    /** Max relative yaw delta (degrees) represented by the outermost yaw bucket. */
    public static final float MAX_YAW_DELTA_DEG = 30.0f;
    /** Max relative pitch delta (degrees) represented by the outermost pitch bucket. */
    public static final float MAX_PITCH_DELTA_DEG = 20.0f;

    /**
     * Maps a yaw bucket index [0, 18] to a signed relative yaw delta in
     * degrees, using a symmetric linear spacing centered on bucket 9
     * (neutral, delta = 0). Buckets are ACTIONS (relative mouse deltas),
     * never absolute target angles - see spec section 18.
     */
    public static float yawBucketToDeltaDegrees(int bucket) {
        int center = (YAW_BUCKETS - 1) / 2; // = 9
        float frac = (bucket - center) / (float) center; // [-1, 1]
        return frac * MAX_YAW_DELTA_DEG;
    }

    public static float pitchBucketToDeltaDegrees(int bucket) {
        int center = (PITCH_BUCKETS - 1) / 2; // = 8
        float frac = (bucket - center) / (float) center; // [-1, 1]
        return frac * MAX_PITCH_DELTA_DEG;
    }

    /** Decomposes a movement action id into forward/strafe unit components in [-1,0,1]. */
    public static int moveForwardComponent(int moveAction) {
        // row 0 = forward(+1), row 1 = neutral(0), row 2 = backward(-1)
        int row = moveAction / 3;
        return row == 0 ? 1 : (row == 2 ? -1 : 0);
    }

    public static int moveStrafeComponent(int moveAction) {
        // col 0 = left(-1), col 1 = neutral(0), col 2 = right(+1)
        int col = moveAction % 3;
        return col == 0 ? -1 : (col == 2 ? 1 : 0);
    }
}
