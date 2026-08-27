package com.example.newgen6.rl;

/**
 * Pure primitive actions only — no hardcoded tactics.
 * Network must learn W-tap, spacing, crit timing itself.
 *
 * Discrete: movement + jump/sprint/sneak/attack combinations
 * Continuous: mouse dx, dy
 */
public final class ActionSpace {

    public static final int IDLE         = 0;
    public static final int FORWARD      = 1;
    public static final int BACK         = 2;
    public static final int LEFT         = 3;
    public static final int RIGHT        = 4;
    public static final int FWD_LEFT     = 5;
    public static final int FWD_RIGHT    = 6;
    public static final int BACK_LEFT    = 7;
    public static final int BACK_RIGHT   = 8;
    public static final int SPRINT_FWD   = 9;
    public static final int JUMP         = 10;
    public static final int JUMP_FWD     = 11;
    public static final int SNEAK        = 12;
    public static final int ATTACK       = 13;
    public static final int ATK_FWD      = 14;
    public static final int ATK_BACK     = 15;
    public static final int ATK_LEFT     = 16;
    public static final int ATK_RIGHT    = 17;
    public static final int ATK_JUMP     = 18;
    public static final int ATK_SPRINT   = 19;
    public static final int ATK_SNEAK    = 20;
    public static final int JUMP_LEFT    = 21;
    public static final int JUMP_RIGHT   = 22;
    public static final int SPRINT_LEFT  = 23;
    public static final int SPRINT_RIGHT = 24;

    public static final int COUNT = 25; // keep in sync with Config.ACTION_DIM

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
        c.mouseDx = mouseDx * lookScale * 1.35f;
        c.mouseDy = mouseDy * lookScale * 1.05f;

        // Each case label appears exactly once
        switch (discrete) {
            case FORWARD -> c.forward = 1f;
            case BACK, ATK_BACK -> c.forward = -1f;
            case LEFT, ATK_LEFT -> c.strafe = 1f;
            case RIGHT, ATK_RIGHT -> c.strafe = -1f;
            case FWD_LEFT -> { c.forward = 1f; c.strafe = 1f; }
            case FWD_RIGHT -> { c.forward = 1f; c.strafe = -1f; }
            case BACK_LEFT -> { c.forward = -1f; c.strafe = 1f; }
            case BACK_RIGHT -> { c.forward = -1f; c.strafe = -1f; }
            case SPRINT_FWD -> { c.forward = 1f; c.sprint = true; }
            case SPRINT_LEFT -> { c.forward = 1f; c.strafe = 1f; c.sprint = true; }
            case SPRINT_RIGHT -> { c.forward = 1f; c.strafe = -1f; c.sprint = true; }
            case JUMP -> c.jump = true;
            case JUMP_FWD -> { c.jump = true; c.forward = 1f; }
            case JUMP_LEFT -> { c.jump = true; c.strafe = 1f; }
            case JUMP_RIGHT -> { c.jump = true; c.strafe = -1f; }
            case SNEAK, ATK_SNEAK -> c.sneak = true;
            case ATK_FWD -> c.forward = 1f;
            case ATK_SPRINT -> { c.forward = 1f; c.sprint = true; }
            case ATK_JUMP -> { c.jump = true; c.forward = 1f; }
            default -> {}
        }

        if (discrete == ATTACK || discrete == ATK_FWD || discrete == ATK_BACK
                || discrete == ATK_LEFT || discrete == ATK_RIGHT
                || discrete == ATK_JUMP || discrete == ATK_SPRINT || discrete == ATK_SNEAK) {
            c.attack = true;
        }
        return c;
    }
}