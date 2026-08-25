
package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.Action;
import com.example.newgen6.rl.RLConstants;
import com.example.newgen6.rl.env.Observation;
import com.example.newgen6.rl.env.ObservationValidator;

import java.util.Arrays;

/**
 * Fixed-capacity rollout storage for recurrent PPO.
 *
 * All arrays use the same timestep dimension. The buffer intentionally
 * owns its arrays so training can reuse memory instead of growing lists
 * indefinitely.
 */
public final class RolloutBuffer {

    private final int capacity;
    private final int observationSize;
    private final int hiddenSize;

    public final float[][] obs;
    public final float[][] hiddenIn;

    public final int[] moveAction;
    public final int[] yawBucket;
    public final int[] pitchBucket;
    public final int[] jumpAction;
    public final int[] sprintAction;
    public final int[] sneakAction;
    public final int[] attackAction;

    public final double[] oldLogProb;
    public final float[] value;
    public final float[] reward;
    public final boolean[] done;

    public final long[] tick;

    private int size;

    public RolloutBuffer() {
        this(
                RLConstants.ROLLOUT_CAPACITY,
                RLConstants.OBSERVATION_SIZE,
                RLConstants.GRU_HIDDEN_SIZE
        );
    }

    public RolloutBuffer(int capacity, int observationSize, int hiddenSize) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0");
        }
        if (observationSize <= 0) {
            throw new IllegalArgumentException("observationSize must be > 0");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be > 0");
        }

        this.capacity = capacity;
        this.observationSize = observationSize;
        this.hiddenSize = hiddenSize;

        obs = new float[capacity][observationSize];
        hiddenIn = new float[capacity][hiddenSize];

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

        tick = new long[capacity];
    }

    public void add(
            Observation observation,
            float[] hiddenState,
            Action action,
            double logProbability,
            float stateValue,
            float stepReward,
            boolean terminal) {

        if (isFull()) {
            throw new IllegalStateException(
                    "RolloutBuffer is full (" + capacity + ")"
            );
        }

        if (observation == null) {
            throw new IllegalArgumentException("observation cannot be null");
        }

        if (hiddenState == null || hiddenState.length != hiddenSize) {
            throw new IllegalArgumentException(
                    "hiddenState size mismatch: got "
                            + (hiddenState == null ? "null" : hiddenState.length)
                            + ", expected " + hiddenSize
            );
        }

        if (action == null) {
            throw new IllegalArgumentException("action cannot be null");
        }

        ObservationValidator.validate(observation);
        action.validate();

        if (!Double.isFinite(logProbability)) {
            throw new IllegalArgumentException(
                    "old log probability must be finite"
            );
        }

        if (!Float.isFinite(stateValue)) {
            throw new IllegalArgumentException(
                    "value must be finite"
            );
        }

        if (!Float.isFinite(stepReward)) {
            throw new IllegalArgumentException(
                    "reward must be finite"
            );
        }

        System.arraycopy(observation.copyValues(), 0, obs[size], 0, observationSize);
        System.arraycopy(hiddenState, 0, hiddenIn[size], 0, hiddenSize);

        moveAction[size] = action.move();
        yawBucket[size] = action.yaw();
        pitchBucket[size] = action.pitch();
        jumpAction[size] = action.jump();
        sprintAction[size] = action.sprint();
        sneakAction[size] = action.sneak();
        attackAction[size] = action.attack();

        oldLogProb[size] = logProbability;
        value[size] = stateValue;
        reward[size] = stepReward;
        done[size] = terminal;
        tick[size] = observation.tick();

        size++;
    }

    public boolean isFull() {
        return size >= capacity;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public int observationSize() {
        return observationSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int lastIndex() {
        if (size == 0) {
            throw new IllegalStateException("RolloutBuffer is empty");
        }
        return size - 1;
    }

    public boolean lastDone() {
        return !isEmpty() && done[lastIndex()];
    }

    public float lastReward() {
        if (isEmpty()) throw new IllegalStateException("RolloutBuffer is empty");
        return reward[lastIndex()];
    }

    public float lastValue() {
        if (isEmpty()) throw new IllegalStateException("RolloutBuffer is empty");
        return value[lastIndex()];
    }

    /**
     * Clears the logical contents while retaining allocated memory.
     */
    public void clear() {
        size = 0;

        for (int i = 0; i < capacity; i++) {
            Arrays.fill(obs[i], 0.0f);
            Arrays.fill(hiddenIn[i], 0.0f);
        }

        Arrays.fill(moveAction, 0);
        Arrays.fill(yawBucket, 0);
        Arrays.fill(pitchBucket, 0);
        Arrays.fill(jumpAction, 0);
        Arrays.fill(sprintAction, 0);
        Arrays.fill(sneakAction, 0);
        Arrays.fill(attackAction, 0);

        Arrays.fill(oldLogProb, 0.0);
        Arrays.fill(value, 0.0f);
        Arrays.fill(reward, 0.0f);
        Arrays.fill(done, false);
        Arrays.fill(tick, 0L);
    }

    /**
     * Validates all stored transitions before a PPO update.
     */
    public void validate() {
        if (size < 0 || size > capacity) {
            throw new IllegalStateException(
                    "Invalid rollout size: " + size
            );
        }

        for (int t = 0; t < size; t++) {
            if (obs[t].length != observationSize) {
                throw new IllegalStateException(
                        "Observation row " + t
                                + " has length " + obs[t].length
                                + ", expected " + observationSize
                );
            }

            if (hiddenIn[t].length != hiddenSize) {
                throw new IllegalStateException(
                        "Hidden row " + t
                                + " has length " + hiddenIn[t].length
                                + ", expected " + hiddenSize
                );
            }

            if (!Double.isFinite(oldLogProb[t])) {
                throw new IllegalStateException(
                        "Non-finite oldLogProb at timestep " + t
                );
            }

            if (!Float.isFinite(value[t])) {
                throw new IllegalStateException(
                        "Non-finite value at timestep " + t
                );
            }

            if (!Float.isFinite(reward[t])) {
                throw new IllegalStateException(
                        "Non-finite reward at timestep " + t
                );
            }

            validateActionIndex(moveAction[t], RLConstants.MOVE_ACTIONS, "move", t);
            validateActionIndex(yawBucket[t], RLConstants.YAW_ACTIONS, "yaw", t);
            validateActionIndex(pitchBucket[t], RLConstants.PITCH_ACTIONS, "pitch", t);
            validateActionIndex(jumpAction[t], RLConstants.JUMP_ACTIONS, "jump", t);
            validateActionIndex(sprintAction[t], RLConstants.SPRINT_ACTIONS, "sprint", t);
            validateActionIndex(sneakAction[t], RLConstants.SNEAK_ACTIONS, "sneak", t);
            validateActionIndex(attackAction[t], RLConstants.ATTACK_ACTIONS, "attack", t);

            for (int j = 0; j < observationSize; j++) {
                if (!Float.isFinite(obs[t][j])) {
                    throw new IllegalStateException(
                            "Non-finite observation at timestep "
                                    + t + ", index " + j
                    );
                }
            }
        }
    }

    private static void validateActionIndex(
            int action,
            int size,
            String name,
            int timestep) {

        if (action < 0 || action >= size) {
            throw new IllegalStateException(
                    "Invalid " + name
                            + " action " + action
                            + " at timestep " + timestep
                            + ", expected [0, " + (size - 1) + "]"
            );
        }
    }
}