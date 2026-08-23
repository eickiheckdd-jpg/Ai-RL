package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;

import java.util.List;

public class NewGen6RLMod implements ModInitializer {
    private static PPOAgent agent;
    private static PPOTrainerThread trainerThread;
    private static Thread trainerWorker;
    private static RolloutBuffer rolloutBuffer;
    private static RewardShaper rewardShaper;

    private static float totalEpisodeReward = 0.0f;
    private static int episodeLength = 0;

    @Override
    public void onInitialize() {
        // Init Agent - It will automatically load the brain from RLConfig.MODEL_FILE if it exists
        agent = new PPOAgent(RLConfig.MODEL_FILE);
        rolloutBuffer = new RolloutBuffer(RLConfig.N_STEPS);
        rewardShaper = new RewardShaper();

        trainerThread = new PPOTrainerThread(agent);
        trainerWorker = new Thread(trainerThread, "PPO-Trainer-Worker");
        trainerWorker.start();

        KeyInputHandler.register();

        // Save Brain on Server Stop
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            System.out.println("[NewGen6] Server shutting down, halting trainer and saving brain...");
            trainerThread.stop();
            agent.saveBrain(RLConfig.MODEL_FILE);
        });

        ServerTickEvents.END_SERVER_TICK.register(server -> {
            PlayerEntity player = server.getOverworld().getPlayers().stream().findFirst().orElse(null);
            if (player == null) return;

            LivingEntity target = server.getOverworld().getEntitiesByClass(
                LivingEntity.class, 
                player.getBoundingBox().expand(20.0), 
                e -> e != player && e.isAlive()
            ).stream().findFirst().orElse(null);

            float[] obs = CombatObservationBuilder.buildObservation(player, target);
            PPOAgent.InferenceResult action = agent.step(obs);

            // Compute Aim Reward using normalized angle offsets (obs[1] = yaw diff, obs[2] = pitch diff)
            float reward = rewardShaper.computeReward(player, target, action.yawDelta(), action.pitchDelta(), obs[1], obs[2], obs[0] == 1.0f);
            totalEpisodeReward += reward;
            episodeLength++;

            boolean done = (target != null && !target.isAlive()) || !player.isAlive() || episodeLength > 1200;

            rolloutBuffer.add(
                obs, 
                new float[]{action.yawDelta(), action.pitchDelta()}, 
                0.0f, 
                reward, 
                action.valueEstimate(), 
                done
            );

            if (rolloutBuffer.isFull()) {
                List<RolloutBuffer.Step> steps = rolloutBuffer.getAndClear();
                trainerThread.enqueueRollout(steps);
            }

            if (RLConfig.agentEnabled) {
                CombatActionExecutor.execute(player, action);
            }

            RLTelemetryBus.update(new RLTelemetryBus.TelemetrySnapshot(
                obs, action.yawDelta(), action.pitchDelta(), action.valueEstimate(), 
                reward, totalEpisodeReward, episodeLength
            ));

            if (done) {
                rewardShaper.reset();
                totalEpisodeReward = 0.0f;
                episodeLength = 0;
            }
        });
    }
}
