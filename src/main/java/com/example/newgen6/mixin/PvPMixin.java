package com.example.newgen6.mixin;

import com.example.newgen6.NewGen6RLMod;
import com.example.newgen6.hud.RlHudOverlay;
import com.example.newgen6.rl.env.ActionSpace;
import com.example.newgen6.rl.env.RewardCalculator;
import com.example.newgen6.rl.env.StateEncoder;
import com.example.newgen6.rl.ppo.ActorCriticNetwork;
import com.example.newgen6.rl.ppo.TrajectoryBuffer;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.Hand;
import net.minecraft.util.math.Box;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.List;

@Mixin(MinecraftClient.class)
public class PvPMixin {

    private final StateEncoder stateEncoder = new StateEncoder();
    private final RewardCalculator rewardCalculator = new RewardCalculator();

    private float[] pendingState = null;
    private int[] pendingActions = null;
    private float pendingLogProb = 0f;
    private float pendingValue = 0f;

    @Inject(method = "tick", at = @At("HEAD"))
    private void newgen6_onClientTick(CallbackInfo ci) {
        MinecraftClient client = (MinecraftClient) (Object) this;
        if (client.world == null || client.player == null || client.isPaused()) return;
        if (!NewGen6RLMod.isEnabled()) return;

        ClientPlayerEntity self = client.player;
        LivingEntity target = findDummyTarget(client, self);

        // 1) Score the action taken on the PREVIOUS tick using the health
        //    deltas observed between then and now, then store the transition.
        if (pendingState != null) {
            float reward = rewardCalculator.step(self, target);
            boolean done = !self.isAlive();

            TrajectoryBuffer buffer = NewGen6RLMod.getBuffer();
            buffer.add(pendingState, pendingActions, reward, pendingValue, pendingLogProb, done);

            // Send real-time reward to HUD
            RlHudOverlay.lastReward = reward;

            if (buffer.isFull()) {
                float bootstrapValue = (self.isAlive() && target != null)
                        ? NewGen6RLMod.getNetwork().act(stateEncoder.encode(self, target)).value
                        : 0f;
                NewGen6RLMod.triggerTrainingAsync(bootstrapValue);
            }
        }

        if (!self.isAlive() || target == null) {
            releaseKeys(client);
            pendingState = null;
            return;
        }

        // 2) Observe the CURRENT state and act.
        float[] state = stateEncoder.encode(self, target);
        ActorCriticNetwork network = NewGen6RLMod.getNetwork();
        ActorCriticNetwork.StepResult step = network.act(state);

        applyAction(client, self, target, step.actions);

        // Feed telemetry data to HUD
        RlHudOverlay.lastValue = step.value;
        RlHudOverlay.lastLogProb = step.logProb;
        RlHudOverlay.lastActionSummary = String.format("M:%d J:%d Y:%d P:%d A:%d", 
            step.actions[0], step.actions[1], step.actions[2], step.actions[3], step.actions[4]);

        pendingState = state;
        pendingActions = step.actions;
        pendingLogProb = step.logProb;
        pendingValue = step.value;
    }

    private void applyAction(MinecraftClient client, ClientPlayerEntity self, LivingEntity target, int[] actions) {
        int move = actions[ActionSpace.GROUP_MOVE];
        boolean forward = move == 1 || move == 5 || move == 6;
        boolean back = move == 2 || move == 7 || move == 8;
        boolean left = move == 3 || move == 5 || move == 7;
        boolean right = move == 4 || move == 6 || move == 8;
        client.options.forwardKey.setPressed(forward);
        client.options.backKey.setPressed(back);
        client.options.leftKey.setPressed(left);
        client.options.rightKey.setPressed(right);

        int jump = actions[ActionSpace.GROUP_JUMP];
        client.options.jumpKey.setPressed(jump == 1);

        float yawDelta = ActionSpace.YAW_DELTAS[actions[ActionSpace.GROUP_YAW]];
        float pitchDelta = ActionSpace.PITCH_DELTAS[actions[ActionSpace.GROUP_PITCH]];
        self.setYaw(self.getYaw() + yawDelta);
        self.setPitch(Math.max(-90f, Math.min(90f, self.getPitch() + pitchDelta)));

        int attack = actions[ActionSpace.GROUP_ATTACK];
        if (attack == 1 && self.distanceTo(target) <= 4.0f && client.interactionManager != null) {
            client.interactionManager.attackEntity(self, target);
            self.swingHand(Hand.MAIN_HAND);
        }
    }

    private void releaseKeys(MinecraftClient client) {
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
    }

    private LivingEntity findDummyTarget(MinecraftClient client, ClientPlayerEntity self) {
        Box searchBox = self.getBoundingBox().expand(16.0);
        List<? extends LivingEntity> candidates = client.world.getEntitiesByClass(
                LivingEntity.class, searchBox,
                e -> e != self
                        && e.isAlive()
                        && !(e instanceof PlayerEntity) 
                        && e.getScoreboardTags().contains("newgen6_dummy")
        );
        LivingEntity nearest = null;
        double best = Double.MAX_VALUE;
        for (LivingEntity e : candidates) {
            double d = e.squaredDistanceTo(self);
            if (d < best) {
                best = d;
                nearest = e;
            }
        }
        return nearest;
    }
}
