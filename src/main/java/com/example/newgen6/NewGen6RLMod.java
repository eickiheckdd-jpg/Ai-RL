package com.example.newgen6;

import com.example.newgen6.rl.ppo.ActorCriticNetwork;
import com.example.newgen6.rl.ppo.PPOConfig;
import com.example.newgen6.rl.ppo.PPOTrainer;
import com.example.newgen6.rl.ppo.TrajectoryBuffer;
import net.fabricmc.api.ClientModInitializer;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class NewGen6RLMod implements ClientModInitializer {

    private static final PPOConfig CONFIG = new PPOConfig();
    private static ActorCriticNetwork network;
    private static volatile TrajectoryBuffer buffer;
    private static PPOTrainer trainer;

    private static final AtomicBoolean enabled = new AtomicBoolean(true);
    private static final AtomicBoolean training = new AtomicBoolean(false);
    private static final ExecutorService trainingExecutor =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "newgen6-ppo-trainer");
                t.setDaemon(true);
                return t;
            });

    private static final File MODEL_FILE = new File("config/newgen6/policy.bin");

    @Override
    public void onInitializeClient() {
        network = new ActorCriticNetwork(System.nanoTime());
        buffer = new TrajectoryBuffer(CONFIG.bufferSize);
        trainer = new PPOTrainer(network, CONFIG);

        if (MODEL_FILE.exists()) {
            try {
                network.load(MODEL_FILE);
                System.out.println("[NewGen6] Loaded existing policy from " + MODEL_FILE);
            } catch (IOException e) {
                System.err.println("[NewGen6] Failed to load policy, starting fresh: " + e.getMessage());
            }
        }
    }

    public static boolean isEnabled() { return enabled.get(); }
    public static void setEnabled(boolean value) { enabled.set(value); }

    public static ActorCriticNetwork getNetwork() { return network; }
    public static TrajectoryBuffer getBuffer() { return buffer; }

    /**
     * Swaps in a fresh buffer immediately (so the client tick keeps collecting
     * the NEXT trajectory without blocking) and trains on the full one in the
     * background. The network's weights are mutated in place by the trainer;
     * since act() only reads weights during its forward pass, a few ticks of
     * "torn read" while a training step is mid-flight is possible. That's an
     * accepted tradeoff of this async on-policy loop -- if you need strict
     * consistency, wrap act()/trainStep() in a ReadWriteLock instead.
     */
    public static void triggerTrainingAsync(float bootstrapValue) {
        if (!training.compareAndSet(false, true)) return; // a training pass is already running

        TrajectoryBuffer full = buffer;
        buffer = new TrajectoryBuffer(CONFIG.bufferSize);

        trainingExecutor.submit(() -> {
            try {
                long start = System.currentTimeMillis();
                trainer.train(full, bootstrapValue);
                long elapsed = System.currentTimeMillis() - start;
                System.out.println("[NewGen6] PPO update complete in " + elapsed + "ms over "
                        + full.size() + " transitions.");
                try {
                    MODEL_FILE.getParentFile().mkdirs();
                    network.save(MODEL_FILE);
                } catch (IOException e) {
                    System.err.println("[NewGen6] Failed to save policy: " + e.getMessage());
                }
            } catch (Throwable t) {
                System.err.println("[NewGen6] Training step threw an exception:");
                t.printStackTrace();
            } finally {
                training.set(false);
            }
        });
    }
}