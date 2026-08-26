package com.example.newgen6.rl;

public class RolloutBuffer {
    public final int capacity;
    public int size = 0;

    // Pre-allocated flat arrays to eliminate GC overhead
    public final float[][] states;
    public final float[][] actions;
    public final float[] rewards;
    public final float[] values;
    public final float[] logProbs;
    public final boolean[] dones;

    public RolloutBuffer(int capacity, int stateSize, int actionSize) {
        this.capacity = capacity;
        this.states = new float[capacity][stateSize];
        this.actions = new float[capacity][actionSize];
        this.rewards = new float[capacity];
        this.values = new float[capacity];
        this.logProbs = new float[capacity];
        this.dones = new boolean[capacity];
    }

    public void store(float[] state, float[] action, float reward, float value, float logProb, boolean done) {
        if (size >= capacity) return; // Buffer is full, wait for PPO update

        System.arraycopy(state, 0, states[size], 0, state.length);
        System.arraycopy(action, 0, actions[size], 0, action.length);
        rewards[size] = reward;
        values[size] = value;
        logProbs[size] = logProb;
        dones[size] = done;
        
        size++;
    }

    public void clear() {
        size = 0; // Reset pointer without destroying memory
    }
}
