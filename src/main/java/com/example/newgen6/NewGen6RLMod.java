package com.example.newgen6;

import com.example.newgen6.client.AiControlState;
import com.example.newgen6.client.AiInputApplier;
import com.example.newgen6.hud.CombatHud;
import com.example.newgen6.rl.env.ActionSpace;
import com.example.newgen6.rl.env.CombatEnv;
import com.example.newgen6.rl.env.ObservationSchema;
import com.example.newgen6.rl.env.RewardFunction;
import com.example.newgen6.rl.nn.AdamOptimizer;
import com.example.newgen6.rl.nn.Categorical;
import com.example.newgen6.rl.nn.PolicyValueNetwork;
import com.example.newgen6.rl.ppo.PPOTrainer;
import com.example.newgen6.rl.ppo.RolloutBuffer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;

import java.util.Random;

/**
 * Client entrypoint (matches fabric.mod.json's declared entrypoint exactly:
 * com.example.newgen6.NewGen6RLMod). Wires the whole pipeline together every
 * tick:
 *
 *   CombatEnv.collect() -> float[229]
 *        -> PolicyValueNetwork.forward(obs, h_prev) -> action distributions + value
 *        -> sample action (stochastic during training, argmax if evaluating)
 *        -> AiControlState / AiInputApplier -> Minecraft's normal input path
 *        -> RolloutBuffer.add(...)
 *        -> when the buffer fills: PPOTrainer.update(...)
 */
public final class NewGen6RLMod implements ClientModInitializer {

    // Random source for weight init AND action sampling. NOT seeded from any
    // pretrained/external source - spec section 7 (start from scratch).
    private final Random rng = new Random();

    private PolicyValueNetwork network;
    private AdamOptimizer optimizer;
    private PPOTrainer trainer;
    private RolloutBuffer rolloutBuffer;
    private CombatEnv env;

    private float[] hiddenState;
    private final float[] obsScratch = new float[ObservationSchema.OBSERVATION_SIZE];

    public static final int ROLLOUT_CAPACITY = 2048; // ticks (~102s of gameplay at 20tps)

    // last-tick bookkeeping for reward computation
    private float lastHealth = -1f;
    private float lastOpponentHealth = -1f;

    @Override
    public void onInitializeClient() {
        ObservationSchema.validate(); // fail loudly at boot if the 229-feature ABI is ever broken

        network = new PolicyValueNetwork(rng); // random init only, no pretrained weights
        optimizer = new AdamOptimizer(3e-4f);
        network.registerWith(optimizer);
        trainer = new PPOTrainer(network, optimizer, rng);
        rolloutBuffer = new RolloutBuffer(ROLLOUT_CAPACITY);
        env = new CombatEnv();
        hiddenState = network.initialHiddenState();

        ClientTickEvents.START_CLIENT_TICK.register(this::onClientTickStart);

        CombatHud.register(() -> lastStats, () -> AiControlState.isAiControlEnabled());
    }

    private volatile PPOTrainer.UpdateStats lastStats = new PPOTrainer.UpdateStats();

    private void onClientTickStart(MinecraftClient mc) {
        ClientPlayerEntity player = mc.player;
        if (player == null || mc.world == null) return;
        if (!AiControlState.isAiControlEnabled()) return;

        env.tickCounters();

        LivingEntity opponent = findOpponent(mc, player);
        env.setOpponent(opponent);

        // 1) Observe (one tick == float[229], see ObservationSchema).
        env.collect(obsScratch);
        float[] hPrev = hiddenState.clone();

        // 2) Policy forward pass (h_prev encodes the recurrent ~200-tick context).
        PolicyValueNetwork.Output out = network.forward(obsScratch, hPrev, null);
        hiddenState = out.h;

        // 3) Sample the factorized action (stochastic - this is training, not eval).
        int move = Categorical.sample(out.moveLogits, rng);
        int yaw = Categorical.sample(out.yawLogits, rng);
        int pitch = Categorical.sample(out.pitchLogits, rng);
        int jump = Categorical.sample(out.jumpLogits, rng);
        int sprint = Categorical.sample(out.sprintLogits, rng);
        int sneak = Categorical.sample(out.sneakLogits, rng);
        int attack = Categorical.sample(out.attackLogits, rng);

        double logProb = com.example.newgen6.rl.ppo.PPOMath.jointLogProb(out, move, yaw, pitch, jump, sprint, sneak, attack);

        // 4) Apply the action through Minecraft's NORMAL input path (no teleport/god-mode - spec 21/23).
        AiControlState.setPendingMovementInput(new AiControlState.PendingInput(
            ActionSpace.moveForwardComponent(move), ActionSpace.moveStrafeComponent(move),
            jump == 1, sneak == 1));
        AiInputApplier.applySprint(player, sprint == 1);
        float yawDeltaDeg = ActionSpace.yawBucketToDeltaDegrees(yaw);
        float pitchDeltaDeg = ActionSpace.pitchBucketToDeltaDegrees(pitch);
        AiInputApplier.applyAim(player, yaw, pitch);
        boolean attackIssued = attack == 1 && opponent != null && AiInputApplier.applyAttack(player, opponent);

        env.recordAction(move, jump == 1, sprint == 1, sneak == 1, attackIssued, yawDeltaDeg, pitchDeltaDeg);

        CombatHud.publish(out.yawLogits, out.pitchLogits, hiddenState, moveLabel(move));

        // 5) Reward from REAL resulting state deltas (health changes since last tick).
        float currentHealth = player.getHealth();
        float damageTaken = lastHealth >= 0 ? Math.max(0, lastHealth - currentHealth) : 0f;
        float opponentHealth = opponent != null ? opponent.getHealth() : 0f;
        float damageDealt = (opponent != null && lastOpponentHealth >= 0) ? Math.max(0, lastOpponentHealth - opponentHealth) : 0f;
        boolean died = currentHealth <= 0f;
        boolean killed = opponent != null && !opponent.isAlive() && lastOpponentHealth > 0f;
        boolean centered = obsScratch[ObservationSchema.TARGET_IS_CENTERED] > 0.5f;

        if (damageTaken > 0) env.onDamageTaken(damageTaken);
        if (attackIssued) env.onDamageDealt(damageDealt, damageDealt > 0);

        float reward = RewardFunction.compute(damageDealt, damageTaken, killed, died, centered);
        lastHealth = currentHealth;
        lastOpponentHealth = opponentHealth;

        boolean episodeDone = died || killed;

        // 6) Store the transition.
        rolloutBuffer.add(obsScratch, hPrev, move, yaw, pitch, jump, sprint, sneak, attack,
            logProb, out.value, reward, episodeDone);

        if (episodeDone) {
            env.resetEpisode();
            hiddenState = network.initialHiddenState(); // reset recurrent context at episode boundaries
        }

        // 7) Train once the rollout buffer is full.
        if (rolloutBuffer.isFull()) {
            float bootstrapValue = episodeDone ? 0f : out.value;
            lastStats = trainer.update(rolloutBuffer, bootstrapValue);
            rolloutBuffer.clear();
        }
    }

    private static String moveLabel(int move) {
        return switch (move) {
            case ActionSpace.MOVE_HOLD -> "HOLD";
            case ActionSpace.MOVE_FORWARD -> "FORWARD";
            case ActionSpace.MOVE_BACKWARD -> "BACKWARD";
            case ActionSpace.MOVE_LEFT -> "LEFT";
            case ActionSpace.MOVE_RIGHT -> "RIGHT";
            case ActionSpace.MOVE_FWD_LEFT -> "FWD-LEFT";
            case ActionSpace.MOVE_FWD_RIGHT -> "FWD-RIGHT";
            case ActionSpace.MOVE_BACK_LEFT -> "BACK-LEFT";
            case ActionSpace.MOVE_BACK_RIGHT -> "BACK-RIGHT";
            default -> "?";
        };
    }

    private LivingEntity findOpponent(MinecraftClient mc, ClientPlayerEntity self) {
        if (mc.world == null) return null;
        LivingEntity closest = null;
        double closestDist = Double.MAX_VALUE;
        for (var entity : mc.world.getEntities()) {
            if (!(entity instanceof LivingEntity living) || entity == self) continue;
            if (!living.isAlive()) continue;
            double d = self.squaredDistanceTo(entity);
            if (d < closestDist && d < ObservationSchema.MAX_RANGE * ObservationSchema.MAX_RANGE) {
                closestDist = d;
                closest = living;
            }
        }
        return closest;
    }
}