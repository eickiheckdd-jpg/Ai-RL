package com.example.newgen6;

import java.util.Random;

public class ReplayBuffer {
    private final Transition[] buffer;
    private final int capacity;
    private int head = 0;
    private int size = 0;
    private final Random random = new Random();

    public ReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new Transition[capacity];
    }

    public synchronized void add(float[] state, int action, float reward, float[] nextState, boolean done) {
        buffer[head] = new Transition(state, action, reward, nextState, done);
        head = (head + 1) % capacity;
        if (size < capacity) size++;
    }

    public synchronized Transition[] sample(int batchSize) {
        int count = Math.min(batchSize, size);
        Transition[] batch = new Transition[count];
        for (int i = 0; i < count; i++) {
            batch[i] = buffer[random.nextInt(size)];
        }
        return batch;
    }

    public synchronized int size() {
        return size;
    }

    public synchronized void clear() {
        head = 0;
        size = 0;
    }
}
