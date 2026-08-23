package com.example.newgen6;

import java.util.LinkedList;
import java.util.Random;

public class ContinuousReplayBuffer {
    private final int capacity;
    private final LinkedList<ContinuousTransition> buffer;
    private final Random random;

    public ContinuousReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new LinkedList<>();
        this.random = new Random();
    }

    public void add(float[] state, float[] action, float reward, float[] nextState, boolean done) {
        if (buffer.size() >= capacity) {
            buffer.removeFirst();
        }
        buffer.addLast(new ContinuousTransition(
            state.clone(), action.clone(), reward, nextState.clone(), done
        ));
    }

    public ContinuousTransition[] sample(int batchSize) {
        ContinuousTransition[] batch = new ContinuousTransition[batchSize];
        for (int i = 0; i < batchSize; i++) {
            batch[i] = buffer.get(random.nextInt(buffer.size()));
        }
        return batch;
    }

    public int size() {
        return buffer.size();
    }
}
