package com.example.newgen6.rl;

import com.example.newgen6.rl.nn.Categorical;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.util.Random;

/**
 * Samples a complete multi-head action from policy logits.
 * Pure RL logic: it does not apply the action to Minecraft.
 */
public final class ActionSampler {

    private final Random random;

    public ActionSampler(Random random) {
        if (random == null) {
            throw new IllegalArgumentException("random cannot be null");
        }
        this.random = random;
    }

    public Sample sample(PolicyValueNetwork.Output output) {
        if (output == null) {
            throw new IllegalArgumentException("output cannot be null");
        }

        int move = Categorical.sample(output.moveLogits, random);
        int yaw = Categorical.sample(output.yawLogits, random);
        int pitch = Categorical.sample(output.pitchLogits, random);
        int jump = Categorical.sample(output.jumpLogits, random);
        int sprint = Categorical.sample(output.sprintLogits, random);
        int sneak = Categorical.sample(output.sneakLogits, random);
        int attack = Categorical.sample(output.attackLogits, random);

        Action action = new Action(
                move, yaw, pitch, jump, sprint, sneak, attack
        );

        double logProb =
                Categorical.logProb(output.moveLogits, move)
                + Categorical.logProb(output.yawLogits, yaw)
                + Categorical.logProb(output.pitchLogits, pitch)
                + Categorical.logProb(output.jumpLogits, jump)
                + Categorical.logProb(output.sprintLogits, sprint)
                + Categorical.logProb(output.sneakLogits, sneak)
                + Categorical.logProb(output.attackLogits, attack);

        return new Sample(action, logProb, output.value);
    }

    public record Sample(
            Action action,
            double logProb,
            double value
    ) {}
}