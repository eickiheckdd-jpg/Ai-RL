package com.example.newgen6.rl;

import com.example.newgen6.rl.env.Observation;
import com.example.newgen6.rl.nn.AdamOptimizer;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.util.Random;

/**
 * Owns the policy/value network and recurrent state.
 *
 * This class deliberately separates inference/training from Minecraft
 * input execution. The environment supplies observations and receives
 * an Action object; another layer decides how/when to apply it.
 */
public final class RLAgent {

    private final PolicyValueNetwork network;
    private final ActionSampler sampler;

    private float[] hiddenState;
    private boolean enabled;

    public RLAgent(Random random) {
        if (random == null) {
            throw new IllegalArgumentException("random cannot be null");
        }

        this.network = new PolicyValueNetwork(random);

        AdamOptimizer optimizer = new AdamOptimizer(
                RLConstants.LEARNING_RATE,
                RLConstants.ADAM_BETA1,
                RLConstants.ADAM_BETA2,
                RLConstants.ADAM_EPSILON
        );

        network.registerWith(optimizer);
        this.sampler = new ActionSampler(random);
        this.hiddenState =
                new float[RLConstants.GRU_HIDDEN_SIZE];
    }

    public ActionSampler.Sample act(Observation observation) {
        if (observation == null) {
            throw new IllegalArgumentException("observation cannot be null");
        }

        float[] encoded = observation.encode();
        if (encoded.length != RLConstants.OBSERVATION_SIZE) {
            throw new IllegalStateException(
                    "Observation ABI mismatch: "
                            + encoded.length
                            + " != "
                            + RLConstants.OBSERVATION_SIZE
            );
        }

        PolicyValueNetwork.Output output =
                network.forward(encoded, hiddenState, null);

        hiddenState = output.hiddenOut;

        return sampler.sample(output);
    }

    public void resetRecurrentState() {
        hiddenState =
                new float[RLConstants.GRU_HIDDEN_SIZE];
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public PolicyValueNetwork network() {
        return network;
    }
}