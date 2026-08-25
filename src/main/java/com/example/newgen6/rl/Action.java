package com.example.newgen6.rl;

/**
 * Immutable action selected by the policy.
 *
 * This class only represents an action. It does not execute or apply
 * the action to Minecraft.
 */
public final class Action {

    public static final int MOVE_ACTIONS = RLConstants.MOVE_ACTIONS;
    public static final int YAW_ACTIONS = RLConstants.YAW_ACTIONS;
    public static final int PITCH_ACTIONS = RLConstants.PITCH_ACTIONS;
    public static final int JUMP_ACTIONS = RLConstants.JUMP_ACTIONS;
    public static final int SPRINT_ACTIONS = RLConstants.SPRINT_ACTIONS;
    public static final int SNEAK_ACTIONS = RLConstants.SNEAK_ACTIONS;
    public static final int ATTACK_ACTIONS = RLConstants.ATTACK_ACTIONS;

    private final int move;
    private final int yaw;
    private final int pitch;
    private final int jump;
    private final int sprint;
    private final int sneak;
    private final int attack;

    public Action(
            int move,
            int yaw,
            int pitch,
            int jump,
            int sprint,
            int sneak,
            int attack) {

        validate(move, MOVE_ACTIONS, "move");
        validate(yaw, YAW_ACTIONS, "yaw");
        validate(pitch, PITCH_ACTIONS, "pitch");
        validate(jump, JUMP_ACTIONS, "jump");
        validate(sprint, SPRINT_ACTIONS, "sprint");
        validate(sneak, SNEAK_ACTIONS, "sneak");
        validate(attack, ATTACK_ACTIONS, "attack");

        this.move = move;
        this.yaw = yaw;
        this.pitch = pitch;
        this.jump = jump;
        this.sprint = sprint;
        this.sneak = sneak;
        this.attack = attack;
    }

    public static Action safeDefault() {
        return new Action(
                0,
                YAW_ACTIONS / 2,
                PITCH_ACTIONS / 2,
                0,
                0,
                0,
                0
        );
    }

    public int move() {
        return move;
    }

    public int yaw() {
        return yaw;
    }

    public int pitch() {
        return pitch;
    }

    public int jump() {
        return jump;
    }

    public int sprint() {
        return sprint;
    }

    public int sneak() {
        return sneak;
    }

    public int attack() {
        return attack;
    }

    public void validate() {
        validate(move, MOVE_ACTIONS, "move");
        validate(yaw, YAW_ACTIONS, "yaw");
        validate(pitch, PITCH_ACTIONS, "pitch");
        validate(jump, JUMP_ACTIONS, "jump");
        validate(sprint, SPRINT_ACTIONS, "sprint");
        validate(sneak, SNEAK_ACTIONS, "sneak");
        validate(attack, ATTACK_ACTIONS, "attack");
    }

    private static void validate(int value, int size, String name) {
        if (value < 0 || value >= size) {
            throw new IllegalArgumentException(
                    name + " action " + value
                            + " outside [0, " + (size - 1) + "]"
            );
        }
    }

    @Override
    public String toString() {
        return "Action{" +
                "move=" + move +
                ", yaw=" + yaw +
                ", pitch=" + pitch +
                ", jump=" + jump +
                ", sprint=" + sprint +
                ", sneak=" + sneak +
                ", attack=" + attack +
                '}';
    }
}