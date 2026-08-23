package com.example.newgen6;

public class Transition {
    public final float[] state;
    public final float[] nextState;
    public int action;
    public float reward;
    public boolean done;

    // Pre-allocate memory once based on the locked state size (16 for your architecture)
    public Transition(int stateSize) {
        this.state = new float[stateSize];
        this.nextState = new float[stateSize];
    }

    public void set(float[] state, int action, float reward, float[] nextState, boolean done) {
        // Zero-allocation lightning copy
        System.arraycopy(state, 0, this.state, 0, state.length);
        System.arraycopy(nextState, 0, this.nextState, 0, nextState.length);
        this.action = action;
        this.reward = reward;
        this.done = done;
    }
}
