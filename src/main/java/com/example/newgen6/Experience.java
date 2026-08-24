package com.example.newgen6;

public class Experience {
    public final double[] state;
    public final int action;
    public final double logProb; // log pi_old(action | state), for the PPO ratio
    public final double value;   // V(state) at collection time
    public final double reward;
    public final boolean done;

    public double advantage;
    public double returnTarget;

    public Experience(double[] state, int action, double logProb, double value, double reward, boolean done) {
        this.state = state;
        this.action = action;
        this.logProb = logProb;
        this.value = value;
        this.reward = reward;
        this.done = done;
    }
}
