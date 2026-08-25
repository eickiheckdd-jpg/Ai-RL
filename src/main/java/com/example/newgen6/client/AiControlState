package com.example.newgen6.client;

/**
 * Small shared state between NewGen6RLMod's tick loop (which decides the
 * action) and PvPMixin (which applies the movement portion of that action
 * through Minecraft's normal keyboard-input path). Not thread-safe beyond
 * "both sides only touch this from the client tick thread", which is the
 * only thread either of them ever runs on.
 */
public final class AiControlState {
    private AiControlState() {}

    private static volatile boolean aiControlEnabled = false;

    public static boolean isAiControlEnabled() { return aiControlEnabled; }
    public static void setAiControlEnabled(boolean enabled) { aiControlEnabled = enabled; }

    /** Movement-only slice of the AI's chosen action, for KeyboardInput to pick up. */
    public static final class PendingInput {
        public final float forward;
        public final float sideways;
        public final boolean jump;
        public final boolean sneak;
        public PendingInput(float forward, float sideways, boolean jump, boolean sneak) {
            this.forward = forward; this.sideways = sideways; this.jump = jump; this.sneak = sneak;
        }
    }

    private static PendingInput pending;

    public static void setPendingMovementInput(PendingInput input) { pending = input; }

    /** Read by PvPMixin at TAIL of KeyboardInput#tick. Left in place (not nulled) so a slow
     *  frame that polls twice doesn't drop the action; it's overwritten every AI tick anyway. */
    public static PendingInput consumePendingMovementInput() { return pending; }
}