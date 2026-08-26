package com.example.newgen6.rl;

import java.util.ArrayList;
import java.util.List;

public class RolloutBuffer {

    public record Transition(
        float[] state,
        int moveAction,
        int yawAction,
        int pitchAction,
        float reward,
        boolean done
    ) {}

    private final List<Transition> batch = new ArrayList<>();

    public void add(float[] state, int moveAction, int yawAction, int pitchAction, float reward, boolean done) {
        batch.add(new Transition(state, moveAction, yawAction, pitchAction, reward, done));
    }

    public List<Transition> getBatch() {
        return batch;
    }

    public void clear() {
        batch.clear();
    }
}
