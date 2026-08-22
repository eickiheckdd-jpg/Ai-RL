package com.example.newgen6;

public class Transition {
    public float[] state;
    public int action;
    public float reward;
    public float[] nextState;
    public boolean done;

    public Transition(float[] state, int action, float reward, float[] nextState, boolean done) {
        set(state, action, reward, nextState, done);
    }

    public void set(float[] state, int action, float reward, float[] nextState, boolean done) {
        this.state = state.clone();
        this.action = action;
        this.reward = reward;
        this.nextState = nextState.clone();
        this.done = done;
    }
}
