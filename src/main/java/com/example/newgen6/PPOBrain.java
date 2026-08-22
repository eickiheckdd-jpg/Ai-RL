package com.example.newgen6.rl;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pure-Java PPO agent. Actor outputs action-logits (softmax applied here),
 * critic outputs a single state-value estimate. Both are 2x64 hidden-unit
 * MLPs as requested.
 *
 * Inference (selectAction) runs synchronously on the client thread and must
 * stay fast (a few matrix multiplies over ~18->64->64->N — sub-millisecond
 * in practice). Training runs asynchronously via ExecutorService so
 * backprop never blocks rendering.
 */
public class PPOBrain {

    private static final int HIDDEN = 64;
    private static final double GAMMA = 0.99;
    private static final double LAMBDA = 0.95;
    private static final double CLIP_EPS = 0.2;
    private static final double LEARNING_RATE = 3e-4;
    private static final double VALUE_COEF = 0.5;
    private static final int PPO_EPOCHS = 4;
    private static final int MINIBATCH_SIZE = 64;

    private final MLP actor;
    private final MLP critic;
    private final Random random = new Random();
    private final ExecutorService trainingExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "newgen6-ppo-trainer");
        t.setDaemon(true);
        return t;
    });
    private final AtomicBoolean trainingInProgress = new AtomicBoolean(false);

    // Telemetry for the HUD — volatile since it's written on the trainer thread
    // and read on the render thread.
    public volatile int epoch = 0;
    public volatile double lastReward = 0.0;
    public volatile double lastLoss = 0.0;
    public volatile int bufferSize = 0;
    public volatile int bufferTarget = 512;

    public PPOBrain() {
        int[] layers = {BotState.FEATURE_COUNT, HIDDEN, HIDDEN, BotAction.COUNT};
        int[] criticLayers = {BotState.FEATURE_COUNT, HIDDEN, HIDDEN, 1};
        this.actor = new MLP(layers, 1234L);
        this.critic = new MLP(criticLayers, 5678L);
    }

    /** Fast synchronous inference used every client tick. */
    public ActionSample selectAction(BotState state, boolean explore) {
        double[] input = state.toVector();
        double[] logits = actor.forward(input);
        double[] probs = softmax(logits);

        int actionIndex;
        if (explore) {
            actionIndex = sampleFromDistribution(probs);
        } else {
            actionIndex = argmax(probs);
        }

        double logProb = Math.log(Math.max(probs[actionIndex], 1e-8));
        double value = critic.forward(input)[0];

        return new ActionSample(BotAction.values()[actionIndex], actionIndex, logProb, value);
    }

    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) max = Math.max(max, l);
        double sum = 0.0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }

    private int sampleFromDistribution(double[] probs) {
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1;
    }

    private int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    /**
     * Kicks off a PPO update on the background thread if the buffer has
     * enough transitions and no update is already running. Safe to call
     * every tick — it no-ops if not ready.
     */
    public void maybeTrainAsync(RolloutBuffer buffer) {
        bufferSize = buffer.size();
        if (bufferSize < bufferTarget) return;
        if (!trainingInProgress.compareAndSet(false, true)) return;

        // Snapshot the buffer contents so the tick thread can keep collecting
        // new transitions into a fresh buffer while training runs.
        double[][] states = buffer.states.toArray(new double[0][]);
        int[] actions = buffer.actions.stream().mapToInt(Integer::intValue).toArray();
        double[] oldLogProbs = buffer.logProbs.stream().mapToDouble(Double::doubleValue).toArray();
        double[] rewards = buffer.rewards.stream().mapToDouble(Double::doubleValue).toArray();
        double[] values = buffer.values.stream().mapToDouble(Double::doubleValue).toArray();
        boolean[] dones = new boolean[buffer.dones.size()];
        for (int i = 0; i < dones.length; i++) dones[i] = buffer.dones.get(i);

        buffer.clear();

        trainingExecutor.submit(() -> {
            try {
                trainOnBatch(states, actions, oldLogProbs, rewards, values, dones);
            } finally {
                trainingInProgress.set(false);
            }
        });
    }

    private void trainOnBatch(double[][] states, int[] actions, double[] oldLogProbs,
                               double[] rewards, double[] values, boolean[] dones) {
        int n = states.length;
        double[] advantages = computeGAE(rewards, values, dones);
        double[] returns = new double[n];
        for (int i = 0; i < n; i++) returns[i] = advantages[i] + values[i];

        normalizeInPlace(advantages);

        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        double runningLoss = 0.0;
        int lossSamples = 0;

        for (int e = 0; e < PPO_EPOCHS; e++) {
            shuffle(indices);
            for (int start = 0; start < n; start += MINIBATCH_SIZE) {
                int end = Math.min(start + MINIBATCH_SIZE, n);
                double[][][] gradWActor = actor.newGradientAccumulatorW();
                double[][] gradBActor = actor.newGradientAccumulatorB();
                double[][][] gradWCritic = critic.newGradientAccumulatorW();
                double[][] gradBCritic = critic.newGradientAccumulatorB();

                int batchCount = end - start;

                for (int b = start; b < end; b++) {
                    int idx = indices[b];
                    double[] state = states[idx];
                    int action = actions[idx];
                    double advantage = advantages[idx];
                    double ret = returns[idx];

                    MLP.ForwardCache actorCache = actor.forwardWithCache(state);
                    double[] logits = actorCache.activations[actorCache.activations.length - 1];
                    double[] probs = softmax(logits);
                    double newLogProb = Math.log(Math.max(probs[action], 1e-8));
                    double ratio = Math.exp(newLogProb - oldLogProbs[idx]);

                    double dObjectiveDRatio;
                    if (advantage >= 0) {
                        dObjectiveDRatio = (ratio <= 1 + CLIP_EPS) ? advantage : 0.0;
                    } else {
                        dObjectiveDRatio = (ratio >= 1 - CLIP_EPS) ? advantage : 0.0;
                    }
                    double dObjectiveDLogProb = dObjectiveDRatio * ratio;
                    double dLossDLogProb = -dObjectiveDLogProb; // minimize -objective

                    // d(logProb of chosen action)/d(logits_j) = 1[j==a] - probs_j
                    double[] outputGradient = new double[probs.length];
                    for (int j = 0; j < probs.length; j++) {
                        double indicator = (j == action) ? 1.0 : 0.0;
                        outputGradient[j] = dLossDLogProb * (indicator - probs[j]);
                    }

                    actor.accumulateGradients(actorCache, outputGradient, gradWActor, gradBActor);

                    // Critic: MSE loss, dLoss/dvalue = 2*(value - return) * VALUE_COEF
                    MLP.ForwardCache criticCache = critic.forwardWithCache(state);
                    double value = criticCache.activations[criticCache.activations.length - 1][0];
                    double dValueLoss = 2.0 * (value - ret) * VALUE_COEF;
                    critic.accumulateGradients(criticCache, new double[]{dValueLoss}, gradWCritic, gradBCritic);

                    runningLoss += Math.abs(dLossDLogProb) + Math.abs(dValueLoss);
                    lossSamples++;
                }

                actor.applyGradients(gradWActor, gradBActor, batchCount, LEARNING_RATE);
                critic.applyGradients(gradWCritic, gradBCritic, batchCount, LEARNING_RATE);
            }
        }

        epoch++;
        lastLoss = lossSamples > 0 ? runningLoss / lossSamples : 0.0;
        lastReward = average(rewards);
    }

    private double[] computeGAE(double[] rewards, double[] values, boolean[] dones) {
        int n = rewards.length;
        double[] advantages = new double[n];
        double lastGae = 0.0;
        for (int t = n - 1; t >= 0; t--) {
            double nextValue = (t == n - 1) ? 0.0 : values[t + 1];
            double nextNonTerminal = dones[t] ? 0.0 : 1.0;
            double delta = rewards[t] + GAMMA * nextValue * nextNonTerminal - values[t];
            lastGae = delta + GAMMA * LAMBDA * nextNonTerminal * lastGae;
            advantages[t] = lastGae;
        }
        return advantages;
    }

    private void normalizeInPlace(double[] arr) {
        double mean = average(arr);
        double variance = 0.0;
        for (double v : arr) variance += (v - mean) * (v - mean);
        variance /= arr.length;
        double std = Math.sqrt(variance) + 1e-8;
        for (int i = 0; i < arr.length; i++) arr[i] = (arr[i] - mean) / std;
    }

    private double average(double[] arr) {
        double sum = 0.0;
        for (double v : arr) sum += v;
        return arr.length > 0 ? sum / arr.length : 0.0;
    }

    private void shuffle(Integer[] arr) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            Integer tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }

    public MLP getActor() { return actor; }
    public MLP getCritic() { return critic; }

    public void shutdown() {
        trainingExecutor.shutdownNow();
    }
}
