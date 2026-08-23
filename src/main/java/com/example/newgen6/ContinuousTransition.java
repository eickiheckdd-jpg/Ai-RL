package com.example.newgen6;

public class ContinuousTransition {
    public final float[] state;
    public final float[] action;
    public final float reward;
    public final float[] nextState;
    public final boolean done;

    public ContinuousTransition(float[] state, float[] action, float reward, float[] nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}
