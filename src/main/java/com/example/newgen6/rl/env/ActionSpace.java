package com.example.newgen6.rl.env;

/** Discrete, multi-headed action space definition. Single source of truth. */
public final class ActionSpace {
    private ActionSpace() {}

    public static final int STATE_SIZE = 78;

    // Movement combos: none, forward, back, left, right, fwd-left, fwd-right, back-left, back-right
    public static final int MOVE_ACTIONS = 9;
    // Yaw turn buckets (degrees/tick), see YAW_DELTAS below
    public static final int YAW_ACTIONS = 9;
    // Pitch turn buckets (degrees/tick), see PITCH_DELTAS below
    public static final int PITCH_ACTIONS = 5;
    // No-op / attack
    public static final int ATTACK_ACTIONS = 2;
    // No-op / jump
    public static final int JUMP_ACTIONS = 2;

    public static final int[] GROUP_SIZES = {
            MOVE_ACTIONS, YAW_ACTIONS, PITCH_ACTIONS, ATTACK_ACTIONS, JUMP_ACTIONS
    };
    public static final int NUM_GROUPS = GROUP_SIZES.length;

    public static final int GROUP_MOVE = 0;
    public static final int GROUP_YAW = 1;
    public static final int GROUP_PITCH = 2;
    public static final int GROUP_ATTACK = 3;
    public static final int GROUP_JUMP = 4;

    public static final float[] YAW_DELTAS = { -60f, -30f, -12f, -4f, 0f, 4f, 12f, 30f, 60f };
    public static final float[] PITCH_DELTAS = { -20f, -6f, 0f, 6f, 20f };
}
