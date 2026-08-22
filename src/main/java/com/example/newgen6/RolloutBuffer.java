package com.example.newgen6.rl;

import java.util.ArrayList;
import java.util.List;

/** Accumulates on-policy transitions until there are enough to run a PPO update. */
public class RolloutBuffer {

    public final List<double[]> states = new ArrayList<>();
    public final List<Integer> actions = new ArrayList<>();
    public final List<Double> logProbs = new ArrayList<>();
    public final List<Double> rewards = new ArrayList<>();
    public final List<Double> values = new ArrayList<>();
    public final List<Boolean> dones = new ArrayList<>();

    public void add(double[] state, int action, double logProb, double reward, double value, boolean done) {
        states.add(state);
        actions.add(action);
        logProbs.add(logProb);
        rewards.add(reward);
        values.add(value);
        dones.add(done);
    }

    public int size() {
        return states.size();
    }

    public void clear() {
        states.clear();
        actions.clear();
        logProbs.clear();
        rewards.clear();
        values.clear();
        dones.clear();
    }
}
