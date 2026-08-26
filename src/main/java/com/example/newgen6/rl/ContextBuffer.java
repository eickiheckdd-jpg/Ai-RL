package com.example.newgen6.rl;

import java.util.Arrays;

public class ContextBuffer {
    public static final int TICKS = 200;
    public static final int FEATURES = 229;
    public static final int TOTAL_FLOATS = TICKS * FEATURES; // 45,800 floats

    private final float[][] ringBuffer = new float[TICKS][FEATURES];
    private final float[] flattenedContext = new float[TOTAL_FLOATS];
    private int head = 0;

    public void push(float[] singleTickObs) {
        // Overwrite the oldest tick slot without creating a new float array
        System.arraycopy(singleTickObs, 0, ringBuffer[head], 0, FEATURES);
        head = (head + 1) % TICKS;
    }

    public float[] getFlattenedContext() {
        // Flatten the temporal matrix chronologically (oldest tick to newest tick)
        int writePos = 0;
        for (int i = 0; i < TICKS; i++) {
            int index = (head + i) % TICKS;
            System.arraycopy(ringBuffer[index], 0, flattenedContext, writePos, FEATURES);
            writePos += FEATURES;
        }
        return flattenedContext;
    }

    public void reset() {
        for (int i = 0; i < TICKS; i++) {
            Arrays.fill(ringBuffer[i], 0.0f);
        }
        head = 0;
    }
}
