package com.example.newgen6.rl;

/**
 * Discrete action space applied directly to the local ClientPlayerEntity.
 * Keep this list stable once you have saved weights — changing the action
 * count invalidates the actor network's output layer shape.
 */
public enum BotAction {
    MOVE_FORWARD,
    MOVE_BACK,
    STRAFE_LEFT,
    STRAFE_RIGHT,
    JUMP,
    ATTACK,
    BLOCK,
    LOOK_LEFT,
    LOOK_RIGHT,
    NO_OP;

    public static final int COUNT = values().length;
}
