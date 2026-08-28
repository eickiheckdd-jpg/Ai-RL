package com.example.newgen6.rl;

import java.util.Arrays;

final class RolloutBuffer {
    final int capacity;
    int size;

    final float[] obs;
    final float[] actions;
    final float[] logProbs;
    final float[] values;
    final float[] rewards;
    final float[] dones;

    RolloutBuffer(int capacity) {
        this.capacity = capacity;

        obs = new float[capacity * AgentConfig.OBS_DIM];
        actions = new float[capacity * AgentConfig.ACTION_DIM];
        logProbs = new float[capacity];
        values = new float[capacity];
        rewards = new float[capacity];
        dones = new float[capacity];
    }

    void add(float[] obsIn, float[] actionIn, float logProb, float value, float reward, boolean done) {
        if (size >= capacity) return;

        int o = size * AgentConfig.OBS_DIM;
        int a = size * AgentConfig.ACTION_DIM;

        System.arraycopy(obsIn, 0, obs, o, AgentConfig.OBS_DIM);
        System.arraycopy(actionIn, 0, actions, a, AgentConfig.ACTION_DIM);

        logProbs[size] = logProb;
        values[size] = value;
        rewards[size] = reward;
        dones[size] = done ? 1.0f : 0.0f;

        size++;
    }

    boolean isFull() {
        return size >= capacity;
    }

    PpoBatch createBatch(float lastValue) {
        PpoBatch b = new PpoBatch();

        b.count = size;
        b.obs = Arrays.copyOf(obs, size * AgentConfig.OBS_DIM);
        b.actions = Arrays.copyOf(actions, size * AgentConfig.ACTION_DIM);
        b.oldLogProbs = Arrays.copyOf(logProbs, size);
        b.rewards = Arrays.copyOf(rewards, size);
        b.values = Arrays.copyOf(values, size);
        b.dones = Arrays.copyOf(dones, size);
        b.lastValue = lastValue;

        return b;
    }

    void reset() {
        size = 0;
    }
}