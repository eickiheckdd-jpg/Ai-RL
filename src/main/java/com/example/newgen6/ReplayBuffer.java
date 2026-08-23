package com.example.newgen6;

import java.util.Random;

public class ReplayBuffer {
    private final Transition[] buffer;
    private final int capacity;
    private int head = 0;
    private int size = 0;
    private final Random random = new Random();

    public ReplayBuffer(int capacity, int stateSize) {
        this.capacity = capacity;
        this.buffer = new Transition[capacity];
        
        // Pre-fill the buffer with empty transition objects to prevent GC spikes later
        for (int i = 0; i < capacity; i++) {
            this.buffer[i] = new Transition(stateSize);
        }
    }

    public synchronized void add(float[] state, int action, float reward, float[] nextState, boolean done) {
        // Overwrite existing memory instead of creating new objects
        buffer[head].set(state, action, reward, nextState, done);
        
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
