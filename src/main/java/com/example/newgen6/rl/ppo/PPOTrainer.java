
package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.RLConstants;
import com.example.newgen6.rl.nn.AdamOptimizer;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Recurrent PPO trainer for the NewGen6 policy/value network.
 *
 * Training occurs on bounded rollout segments so memory consumption stays
 * practical on mobile/Pojav.
 */
public final class PPOTrainer {

    public float gamma = RLConstants.GAMMA;
    public float gaeLambda = RLConstants.GAE_LAMBDA;
    public float clipEpsilon = RLConstants.CLIP_EPSILON;

    public float valueCoef = RLConstants.VALUE_COEF;
    public float entropyCoef = RLConstants.ENTROPY_COEF;
    public float maxGradNorm = RLConstants.MAX_GRAD_NORM;

    public int epochs = RLConstants.PPO_EPOCHS;
    public int segmentLength = RLConstants.SEGMENT_LENGTH;

    private final PolicyValueNetwork network;
    private final AdamOptimizer optimizer;
    private final Random rng;

    public PPOTrainer(
            PolicyValueNetwork network,
            AdamOptimizer optimizer,
            Random rng) {

        if (network == null) {
            throw new IllegalArgumentException("network cannot be null");
        }

        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }

        if (rng == null) {
            throw new IllegalArgumentException("rng cannot be null");
        }

        this.network = network;
        this.optimizer = optimizer;
        this.rng = rng;
    }

    public static final class UpdateStats {
        public double meanPolicyLoss;
        public double meanValueLoss;
        public double meanEntropy;
        public double meanApproxKl;
        public double gradNorm;
        public int samples;
        public int epochs;
        public boolean updated;
    }

    public UpdateStats update(
            RolloutBuffer buffer,
            float bootstrapValue) {

        if (buffer == null) {
            throw new IllegalArgumentException("buffer cannot be null");
        }

        buffer.validate();

        int n = buffer.size();

        UpdateStats stats = new UpdateStats();
        stats.epochs = epochs;

        if (n == 0) {
            return stats;
        }

        if (epochs <= 0) {
            throw new IllegalStateException("epochs must be > 0");
        }

        if (segmentLength <= 0 || segmentLength > n) {
            throw new IllegalStateException(
                    "Invalid segmentLength " + segmentLength
                            + " for rollout size " + n
            );
        }

        float[][] gae =
                PPOMath.computeGAE(
                        slice(buffer.reward, n),
                        slice(buffer.value, n),
                        sliceBool(buffer.done, n),
                        bootstrapValue,
                        gamma,
                        gaeLambda
                );

        float[] advantages = gae[0];
        float[] returns = gae[1];

        PPOMath.normalizeAdvantages(advantages);

        List<Integer> starts = new ArrayList<>();

        for (int start = 0; start < n; start += segmentLength) {
            starts.add(start);
        }

        double sumPolicyLoss = 0.0;
        double sumValueLoss = 0.0;
        double sumEntropy = 0.0;
        double sumKl = 0.0;
        int count = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {

            Collections.shuffle(starts, rng);

            for (int start : starts) {

                int end = Math.min(
                        start + segmentLength,
                        n
                );

                network.zeroGrad();

                List<PolicyValueNetwork.StepCache> caches =
                        new ArrayList<>(end - start);

                List<PolicyValueNetwork.HeadGrads> gradients =
                        new ArrayList<>(end - start);

                for (int t = start; t < end; t++) {

                    float[] hiddenIn = buffer.hiddenIn[t];

                    boolean boundary =
                            t > 0 && buffer.done[t - 1];

                    if (boundary) {
                        hiddenIn =
                                new float[hiddenIn.length];
                    }

                    PolicyValueNetwork.StepCache cache =
                            new PolicyValueNetwork.StepCache();

                    cache.doneMask = boundary;

                    PolicyValueNetwork.Output out =
                            network.forward(
                                    buffer.obs[t],
                                    hiddenIn,
                                    cache
                            );

                    double newLogProb =
                            PPOMath.jointLogProb(
                                    out,
                                    buffer.moveAction[t],
                                    buffer.yawBucket[t],
                                    buffer.pitchBucket[t],
                                    buffer.jumpAction[t],
                                    buffer.sprintAction[t],
                                    buffer.sneakAction[t],
                                    buffer.attackAction[t]
                            );

                    double ratio =
                            Math.exp(
                                    clamp(
                                            newLogProb
                                                    - buffer.oldLogProb[t],
                                            -50.0,
                                            50.0
                                    )
                            );

                    double advantage =
                            advantages[t];

                    double unclipped =
                            ratio * advantage;

                    double clippedRatio =
                            Math.max(
                                    1.0 - clipEpsilon,
                                    Math.min(
                                            1.0 + clipEpsilon,
                                            ratio
                                    )
                            );

                    double clipped =
                            clippedRatio * advantage;

                    double objective =
                            Math.min(
                                    unclipped,
                                    clipped
                            );

                    double policyLoss =
                            -objective;

                    sumPolicyLoss += policyLoss;

                    boolean clippedActive =
                            (advantage >= 0.0 &&
                                    ratio > 1.0 + clipEpsilon)
                            ||
                            (advantage < 0.0 &&
                                    ratio < 1.0 - clipEpsilon);

                    double dObjectiveDLogProb;

                    if (clippedActive) {
                        dObjectiveDLogProb = 0.0;
                    } else {
                        dObjectiveDLogProb =
                                advantage * ratio;
                    }

                    /*
                     * Loss is -objective, so:
                     *
                     * dLoss/d(newLogProb)
                     *     = -dObjective/d(newLogProb)
                     */
                    double dLossDLogProb =
                            -dObjectiveDLogProb;

                    double entropy =
                            PPOMath.jointEntropy(out);

                    sumEntropy += entropy;

                    double approxKl =
                            buffer.oldLogProb[t]
                                    - newLogProb;

                    sumKl += approxKl;

                    double oldValue =
                            buffer.value[t];

                    double targetReturn =
                            returns[t];

                    double unclippedError =
                            out.value - targetReturn;

                    double unclippedLoss =
                            unclippedError
                                    * unclippedError;

                    double clippedValue =
                            oldValue
                                    + clamp(
                                            out.value
                                                    - oldValue,
                                            -clipEpsilon,
                                            clipEpsilon
                                    );

                    double clippedError =
                            clippedValue
                                    - targetReturn;

                    double clippedValueLoss =
                            clippedError
                                    * clippedError;

                    boolean useClippedValue =
                            clippedValueLoss
                                    > unclippedLoss;

                    double valueLoss =
                            valueCoef
                                    * Math.max(
                                            unclippedLoss,
                                            clippedValueLoss
                                    );

                    sumValueLoss += valueLoss;

                    double dValueLossDValue;

                    if (useClippedValue) {
                        /*
                         * If the clipped value is active, the derivative
                         * only flows through the unclipped value when it
                         * is inside the clipping interval.
                         */
                        double deltaValue =
                                out.value - oldValue;

                        boolean insideClip =
                                Math.abs(deltaValue)
                                        <= clipEpsilon;

                        dValueLossDValue =
                                insideClip
                                        ? 2.0 * clippedError
                                        : 0.0;
                    } else {
                        dValueLossDValue =
                                2.0 * unclippedError;
                    }

                    dValueLossDValue *= valueCoef;

                    PolicyValueNetwork.HeadGrads hg =
                            new PolicyValueNetwork.HeadGrads();

                    hg.dMoveLogits =
                            PPOMath.policyLogitGradient(
                                    out.moveLogits,
                                    buffer.moveAction[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dYawLogits =
                            PPOMath.policyLogitGradient(
                                    out.yawLogits,
                                    buffer.yawBucket[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dPitchLogits =
                            PPOMath.policyLogitGradient(
                                    out.pitchLogits,
                                    buffer.pitchBucket[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dJumpLogits =
                            PPOMath.policyLogitGradient(
                                    out.jumpLogits,
                                    buffer.jumpAction[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dSprintLogits =
                            PPOMath.policyLogitGradient(
                                    out.sprintLogits,
                                    buffer.sprintAction[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dSneakLogits =
                            PPOMath.policyLogitGradient(
                                    out.sneakLogits,
                                    buffer.sneakAction[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dAttackLogits =
                            PPOMath.policyLogitGradient(
                                    out.attackLogits,
                                    buffer.attackAction[t],
                                    dLossDLogProb,
                                    entropyCoef
                            );

                    hg.dValue =
                            (float) dValueLossDValue;

                    caches.add(cache);
                    gradients.add(hg);

                    count++;
                }

                network.backwardSegment(
                        caches,
                        gradients
                );

                optimizer.clipGlobalNorm(maxGradNorm);

                stats.gradNorm =
                        optimizer.globalGradNorm();

                optimizer.step();
            }
        }

        stats.meanPolicyLoss =
                sumPolicyLoss / Math.max(1, count);

        stats.meanValueLoss =
                sumValueLoss / Math.max(1, count);

        stats.meanEntropy =
                sumEntropy / Math.max(1, count);

        stats.meanApproxKl =
                sumKl / Math.max(1, count);

        stats.samples = count;
        stats.updated = true;

        return stats;
    }

    private static double clamp(
            double value,
            double min,
            double max) {

        return Math.max(
                min,
                Math.min(max, value)
        );
    }

    private static float[] slice(
            float[] source,
            int length) {

        float[] result =
                new float[length];

        System.arraycopy(
                source,
                0,
                result,
                0,
                length
        );

        return result;
    }

    private static boolean[] sliceBool(
            boolean[] source,
            int length) {

        boolean[] result =
                new boolean[length];

        System.arraycopy(
                source,
                0,
                result,
                0,
                length
        );

        return result;
    }
}