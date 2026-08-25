package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.nn.Adam;
import com.example.newgen6.rl.nn.Dense;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Orchestrates one full PPO update over a filled TrajectoryBuffer:
 * GAE -> advantage normalization -> `epochs` passes of shuffled minibatches,
 * each minibatch accumulating gradients via ActorCriticNetwork.trainStep()
 * then applying a single averaged Adam step.
 *
 * Designed to be called from a background thread (see NewGen6RLMod);
 * nothing in here touches Minecraft/client state.
 */
public class PPOTrainer {

    private final ActorCriticNetwork network;
    private final PPOConfig config;
    private final Adam optimizer;
    private final Random rng = new Random();

    public PPOTrainer(ActorCriticNetwork network, PPOConfig config) {
        this.network = network;
        this.config = config;
        this.optimizer = new Adam(config.learningRate);
    }

    public void train(TrajectoryBuffer buffer, float bootstrapValue) {
        int n = buffer.size();
        if (n == 0) return;

        float[] advantages = new float[n];
        float[] returns = new float[n];
        GAE.compute(buffer.rewards, buffer.values, buffer.dones, bootstrapValue,
                config.gamma, config.lambda, advantages, returns);
        normalize(advantages);

        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) indices.add(i);

        for (int epoch = 0; epoch < config.epochs; epoch++) {
            Collections.shuffle(indices, rng);
            for (int start = 0; start < n; start += config.minibatchSize) {
                int end = Math.min(start + config.minibatchSize, n);
                int batchSize = end - start;

                network.zeroGrad();
                for (int idx = start; idx < end; idx++) {
                    int i = indices.get(idx);
                    network.trainStep(
                            buffer.states[i],
                            buffer.actions[i],
                            buffer.logProbs[i],
                            advantages[i],
                            returns[i],
                            config.clipEps,
                            config.entropyCoef,
                            config.valueCoef
                    );
                }

                Dense[] layers = network.allDenseLayers();
                float scale = 1f / batchSize; // average the accumulated-sum gradients
                for (Dense d : layers) d.scaleGrad(scale);
                optimizer.step(layers);
            }
        }
    }

    private static void normalize(float[] arr) {
        double mean = 0;
        for (float v : arr) mean += v;
        mean /= arr.length;
        double variance = 0;
        for (float v : arr) variance += (v - mean) * (v - mean);
        variance /= arr.length;
        float std = (float) Math.sqrt(variance) + 1e-8f;
        for (int i = 0; i < arr.length; i++) arr[i] = (float) ((arr[i] - mean) / std);
    }
}
