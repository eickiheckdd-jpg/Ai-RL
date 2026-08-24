package com.example.newgen6;

/**
 * Every action here is something a real player could do with a keyboard and
 * a mouse limited to a fixed turn-speed - no snap-aim, no through-wall
 * targeting, no instant teleport-attack. The policy has to learn to combine
 * these primitives (turn, then attack once actually aimed) itself.
 */
public enum ActionType {
    NONE,
    MOVE_FORWARD,
    MOVE_BACKWARD,
    STRAFE_LEFT,
    STRAFE_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
    JUMP,
    ATTACK;

    public static final ActionType[] VALUES = values();
}
