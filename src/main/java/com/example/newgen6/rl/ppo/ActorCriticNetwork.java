package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.env.ActionSpace;
import com.example.newgen6.rl.nn.Dense;
import com.example.newgen6.rl.nn.TanhLayer;

import java.io.*;
import java.util.Random;

/**
 * Shared-trunk Actor-Critic network for a factorized multi-discrete action
 * space: 78-float state -> Dense(128) -> Tanh -> Dense(128) -> Tanh -> {5
 * discrete heads + 1 linear value head}.
 */
public class ActorCriticNetwork {

    private static final int TRUNK_HIDDEN = 128;

    private final Dense trunk1;
    private final TanhLayer act1 = new TanhLayer();
    private final Dense trunk2;
    private final TanhLayer act2 = new TanhLayer();

    private final DiscreteHead[] heads;
    private final Dense valueHead;

    private final Random rng;

    public ActorCriticNetwork(long seed) {
        this.rng = new Random(seed);
        trunk1 = new Dense(ActionSpace.STATE_SIZE, TRUNK_HIDDEN, rng);
        trunk2 = new Dense(TRUNK_HIDDEN, TRUNK_HIDDEN, rng);

        heads = new DiscreteHead[ActionSpace.NUM_GROUPS];
        for (int g = 0; g < ActionSpace.NUM_GROUPS; g++) {
            heads[g] = new DiscreteHead(TRUNK_HIDDEN, ActionSpace.GROUP_SIZES[g], rng);
        }
        valueHead = new Dense(TRUNK_HIDDEN, 1, rng);
    }

    private float[] trunkForward(float[] state) {
        float[] h1 = act1.forward(trunk1.forward(state));
        return act2.forward(trunk2.forward(h1));
    }

    /** One full inference step: sampled actions, summed log-prob, and V(s). */
    public static final class StepResult {
        public final int[] actions = new int[ActionSpace.NUM_GROUPS];
        public float logProb;
        public float value;
    }

    /** Inference for the client tick: forward pass + categorical sampling per head. */
    public StepResult act(float[] state) {
        float[] h = trunkForward(state);
        StepResult result = new StepResult();
        float totalLogProb = 0f;
        for (int g = 0; g < heads.length; g++) {
            heads[g].probs(h);
            int a = heads[g].sample(rng);
            result.actions[g] = a;
            totalLogProb += heads[g].logProb(a);
        }
        result.logProb = totalLogProb;
        result.value = valueHead.forward(h)[0];
        return result;
    }

    /**
     * Recomputes new log-prob / value for a stored (state, actions) sample
     * under the CURRENT weights, and immediately backprops the PPO clipped
     * surrogate loss + entropy bonus + value loss for this single sample.
     * Gradients accumulate into each Dense layer -- call zeroGrad() before a
     * minibatch and Adam.step(allDenseLayers()) after, then scaleGrad(1/batchSize)
     * in between to average rather than sum (see PPOTrainer).
     */
    public void trainStep(float[] state, int[] actions, float oldLogProb, float advantage,
                           float returnTarget, float clipEps, float entropyCoef, float valueCoef) {
        float[] h = trunkForward(state);

        float newLogProb = 0f;
        for (int g = 0; g < heads.length; g++) {
            heads[g].probs(h);
            newLogProb += heads[g].logProb(actions[g]);
        }
        float value = valueHead.forward(h)[0];

        // ---- PPO clipped surrogate: dLoss/dRatio, then chain to dLoss/dNewLogProb ----
        float ratio = (float) Math.exp(newLogProb - oldLogProb);
        float surr1 = ratio * advantage;
        float clippedRatio = clamp(ratio, 1f - clipEps, 1f + clipEps);
        float surr2 = clippedRatio * advantage;

        float dLoss_dRatio;
        if (surr1 <= surr2) {
            // unclipped branch selected -> gradient flows normally
            dLoss_dRatio = -advantage;
        } else if (ratio > (1f - clipEps) && ratio < (1f + clipEps)) {
            // clipped branch selected but ratio is inside the clip range (boundary/tie case)
            dLoss_dRatio = -advantage;
        } else {
            // clipping is actively saturating the objective -> zero gradient, PPO's core trick
            dLoss_dRatio = 0f;
        }
        float coeffPolicy = dLoss_dRatio * ratio; // dRatio/dNewLogProb == ratio

        // ---- Value loss: 0.5 * (V - return)^2 ----
        float dLoss_dValue = valueCoef * (value - returnTarget);

        // ---- Backprop through heads, sum gradient wrt trunk output ----
        float[] gradTrunk = new float[h.length];
        for (int g = 0; g < heads.length; g++) {
            float[] gradFromHead = heads[g].backward(actions[g], coeffPolicy, entropyCoef);
            for (int i = 0; i < gradTrunk.length; i++) gradTrunk[i] += gradFromHead[i];
        }
        float[] gradFromValue = valueHead.backward(new float[]{dLoss_dValue});
        for (int i = 0; i < gradTrunk.length; i++) gradTrunk[i] += gradFromValue[i];

        // ---- Backprop through the shared trunk ----
        float[] gradAct2 = act2.backward(gradTrunk);
        float[] gradH1 = trunk2.backward(gradAct2);
        float[] gradAct1 = act1.backward(gradH1);
        trunk1.backward(gradAct1);
    }

    public void zeroGrad() {
        for (Dense d : allDenseLayers()) d.zeroGrad();
    }

    public Dense[] allDenseLayers() {
        Dense[] arr = new Dense[3 + heads.length];
        arr[0] = trunk1;
        arr[1] = trunk2;
        for (int g = 0; g < heads.length; g++) arr[2 + g] = heads[g].dense;
        arr[arr.length - 1] = valueHead;
        return arr;
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ---- persistence ----
    public void save(File file) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            for (Dense d : allDenseLayers()) d.writeTo(out);
        }
    }

    public void load(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            for (Dense d : allDenseLayers()) d.readFrom(in);
        }
    }
}
