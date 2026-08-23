package com.example.newgen6;

import java.util.ArrayList;
import java.util.List;

public class RolloutBuffer {
    public static class Step {
        public float[] obs;
        public float[] continuousAction;
        public float logProb;
        public float reward;
        public float value;
        public boolean done;

        public Step(float[] obs, float[] continuousAction, float logProb, float reward, float value, boolean done) {
            this.obs = obs;
            this.continuousAction = continuousAction;
            this.logProb = logProb;
            this.reward = reward;
            this.value = value;
            this.done = done;
        }
    }

    private final List<Step> buffer = new ArrayList<>();
    private final int capacity;

    public RolloutBuffer(int capacity) {
        this.capacity = capacity;
    }

    public synchronized void add(float[] obs, float[] continuousAction, float logProb, float reward, float value, boolean done) {
        buffer.add(new Step(obs, continuousAction, logProb, reward, value, done));
    }

    public synchronized boolean isFull() {
        return buffer.size() >= capacity;
    }

    public synchronized List<Step> getAndClear() {
        List<Step> copy = new ArrayList<>(buffer);
        buffer.clear();
        return copy;
    }
}
