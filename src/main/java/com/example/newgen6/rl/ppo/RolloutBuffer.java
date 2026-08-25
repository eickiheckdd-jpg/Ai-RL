package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.env.ObservationSchema;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

/**
 * Fixed-capacity ring buffer of collected (obs, hiddenIn, actions,
 * oldLogProb, value, reward, done) transitions, one entry per Minecraft
 * tick. Pre-allocated primitive arrays only - "do NOT allocate 200 Java
 * objects every tick" (spec section 15) is honored here too: nothing is
 * boxed or allocated during collection, only plain array writes.
 */
public final class RolloutBuffer {

    public final int capacity;
    private int writeIndex = 0;
    private int size = 0;

    public final float[][] obs;        // [capacity][229]
    public final float[][] hiddenIn;   // [capacity][HIDDEN_SIZE] - GRU state BEFORE this tick
    public final int[] moveAction, yawBucket, pitchBucket, jumpAction, sprintAction, sneakAction, attackAction;
    public final double[] oldLogProb;  // joint log-prob at collection time (see PPOMath#jointLogProb)
    public final float[] value;        // V(s_t) estimated at collection time
    public final float[] reward;
    public final boolean[] done;

    public RolloutBuffer(int capacity) {
        this.capacity = capacity;
        obs = new float[capacity][ObservationSchema.OBSERVATION_SIZE];
        hiddenIn = new float[capacity][PolicyValueNetwork.HIDDEN_SIZE];
        moveAction = new int[capacity];
        yawBucket = new int[capacity];
        pitchBucket = new int[capacity];
        jumpAction = new int[capacity];
        sprintAction = new int[capacity];
        sneakAction = new int[capacity];
        attackAction = new int[capacity];
        oldLogProb = new double[capacity];
        value = new float[capacity];
        reward = new float[capacity];
        done = new boolean[capacity];
    }

    public void add(float[] obsT, float[] hiddenInT, int move, int yaw, int pitch, int jump, int sprint,
                     int sneak, int attack, double logProb, float valueT, float rewardT, boolean isDone) {
        int i = writeIndex;
        System.arraycopy(obsT, 0, obs[i], 0, ObservationSchema.OBSERVATION_SIZE);
        System.arraycopy(hiddenInT, 0, hiddenIn[i], 0, PolicyValueNetwork.HIDDEN_SIZE);
        moveAction[i] = move; yawBucket[i] = yaw; pitchBucket[i] = pitch;
        jumpAction[i] = jump; sprintAction[i] = sprint; sneakAction[i] = sneak; attackAction[i] = attack;
        oldLogProb[i] = logProb; value[i] = valueT; reward[i] = rewardT; done[i] = isDone;

        writeIndex = (writeIndex + 1) % capacity;
        if (size < capacity) size++;
    }

    public int size() { return size; }
    public boolean isFull() { return size == capacity; }
    public void clear() { size = 0; writeIndex = 0; }
}
