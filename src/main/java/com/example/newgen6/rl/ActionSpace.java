package com.example.newgen6.rl;

/**
 * Hybrid action space for HT3 Sword.
 * Movement/jump/sprint/sneak/attack = discrete (masked)
 * Aim = pure continuous mouse deltas (no discrete look actions)
 */
public final class ActionSpace {

    public static final int DISCRETE = Config.ACTION_DIM;
    public static final int CONTINUOUS = 2;

    public static final int IDLE = 0;
    public static final int FORWARD = 1;
    public static final int BACK = 2;
    public static final int LEFT = 3;
    public static final int RIGHT = 4;
    public static final int FWD_LEFT = 5;
    public static final int FWD_RIGHT = 6;
    public static final int BACK_LEFT = 7;
    public static final int BACK_RIGHT = 8;
    public static final int SPRINT_FWD = 9;
    public static final int JUMP = 10;
    public static final int JUMP_FWD = 11;
    public static final int SNEAK = 12;
    public static final int ATTACK = 13;
    public static final int ATTACK_FWD = 14;
    public static final int ATTACK_BACK = 15;
    public static final int ATTACK_LEFT = 16;
    public static final int ATTACK_RIGHT = 17;
    public static final int ATTACK_JUMP = 18;
    public static final int ATTACK_SPRINT = 19;
    public static final int CRIT_ATTEMPT = 20;
    public static final int STRAFE_LEFT_ATK = 21;
    public static final int STRAFE_RIGHT_ATK = 22;
    public static final int HOLD_DIST = 23;
    public static final int CLOSE_IN = 24;
    public static final int BAIT_BACK = 25;

    public static final class Control {
        public float forward;
        public float strafe;
        public boolean jump;
        public boolean sprint;
        public boolean sneak;
        public boolean attack;
        public float mouseDx;
        public float mouseDy;
        public float confidence;
    }

    public static Control decode(int discrete, float mouseDx, float mouseDy,
                                 float lookScale, float confidence) {
        Control c = new Control();
        c.confidence = confidence;
        c.mouseDx = mouseDx * lookScale * 4.5f;
        c.mouseDy = mouseDy * lookScale * 3.2f;

        switch (discrete) {
            case FORWARD, SPRINT_FWD, ATTACK_FWD, CLOSE_IN -> {
                c.forward = 1f;
                if (discrete == SPRINT_FWD || discrete == ATTACK_FWD) c.sprint = true;
            }
            case BACK, ATTACK_BACK, BAIT_BACK -> c.forward = -1f;
            case LEFT, STRAFE_LEFT_ATK -> c.strafe = 1f;
            case RIGHT, STRAFE_RIGHT_ATK -> c.strafe = -1f;
            case FWD_LEFT -> { c.forward = 1f; c.strafe = 1f; }
            case FWD_RIGHT -> { c.forward = 1f; c.strafe = -1f; }
            case BACK_LEFT -> { c.forward = -1f; c.strafe = 1f; }
            case BACK_RIGHT -> { c.forward = -1f; c.strafe = -1f; }
            case JUMP, JUMP_FWD, ATTACK_JUMP, CRIT_ATTEMPT -> {
                c.jump = true;
                if (discrete != JUMP) c.forward = 1f;
            }
            case SNEAK -> c.sneak = true;
            default -> {}
        }
        if ((discrete >= ATTACK && discrete <= STRAFE_RIGHT_ATK) || discrete == CRIT_ATTEMPT) {
            c.attack = true;
        }
        return c;
    }
}
