package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.env.ActionSpace;

/**
 * Fixed-capacity in-memory rollout buffer. Sized for 2048 ticks by default
 * (see PPOConfig.bufferSize). Thread-safety note: add()/isFull()/size() are
 * synchronized because add() is called from the client thread while a
 * background thread may be reading a *different* (already-swapped-out)
 * buffer instance -- see NewGen6RLMod.triggerTrainingAsync for the
 * swap-and-flush pattern that keeps those two buffers from ever being the
 * same object at the same time.
 */
public class TrajectoryBuffer {

    public final int capacity;
    public final float[][] states;
    public final int[][] actions;
    public final float[] rewards;
    public final float[] values;
    public final float[] logProbs;
    public final boolean[] dones;

    private int size = 0;

    public TrajectoryBuffer(int capacity) {
        this.capacity = capacity;
        states = new float[capacity][ActionSpace.STATE_SIZE];
        actions = new int[capacity][ActionSpace.NUM_GROUPS];
        rewards = new float[capacity];
        values = new float[capacity];
        logProbs = new float[capacity];
        dones = new boolean[capacity];
    }

    public synchronized boolean add(float[] state, int[] action, float reward, float value, float logProb, boolean done) {
        if (size >= capacity) return false;
        System.arraycopy(state, 0, states[size], 0, ActionSpace.STATE_SIZE);
        System.arraycopy(action, 0, actions[size], 0, ActionSpace.NUM_GROUPS);
        rewards[size] = reward;
        values[size] = value;
        logProbs[size] = logProb;
        dones[size] = done;
        size++;
        return true;
    }

    public synchronized boolean isFull() { return size >= capacity; }
    public synchronized int size() { return size; }
    public synchronized void clear() { size = 0; }
}
