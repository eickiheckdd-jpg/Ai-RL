package com.example.newgen6.rl.ppo;

import com.example.newgen6.rl.nn.AdamOptimizer;
import com.example.newgen6.rl.nn.Categorical;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class PPOTrainer {

    public float gamma = 0.99f, gaeLambda = 0.95f, clipEpsilon = 0.2f;
    public float valueCoef = 0.5f, entropyCoef = 0.01f, maxGradNorm = 0.5f;
    public int epochs = 4, segmentLength = 32;

    private final PolicyValueNetwork network;
    private final AdamOptimizer optimizer;
    private final Random rng;

    public PPOTrainer(PolicyValueNetwork network, AdamOptimizer optimizer, Random rng) {
        this.network = network;
        this.optimizer = optimizer;
        this.rng = rng;
    }

    public static final class UpdateStats {
        public double meanPolicyLoss, meanValueLoss, meanEntropy, meanApproxKl, gradNorm;
    }

    public UpdateStats update(RolloutBuffer buf, float bootstrapValue) {
        int n = buf.size();
        float[][] gae = PPOMath.computeGAE(
            slice(buf.reward, n), slice(buf.value, n), sliceBool(buf.done, n), bootstrapValue, gamma, gaeLambda);
        float[] advantages = gae[0];
        float[] returns = gae[1];

        normalizeInPlace(advantages);

        List<Integer> starts = new ArrayList<>();
        for (int s = 0; s < n; s += segmentLength) starts.add(s);

        UpdateStats stats = new UpdateStats();
        double sumPolicyLoss = 0, sumValueLoss = 0, sumEntropy = 0, sumKl = 0;
        int updateCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(starts, rng);
            for (int start : starts) {
                int end = Math.min(start + segmentLength, n);
                network.zeroGrad();

                List<PolicyValueNetwork.StepCache> caches = new ArrayList<>();
                List<PolicyValueNetwork.HeadGrads> grads = new ArrayList<>();

                for (int t = start; t < end; t++) {
                    PolicyValueNetwork.StepCache cache = new PolicyValueNetwork.StepCache();

                    // Zero out GRU state if crossing an episode boundary
                    float[] hIn = buf.hiddenIn[t];
                    boolean isTerminalBoundary = (t > 0 && buf.done[t - 1]);
                    if (isTerminalBoundary) {
                        hIn = new float[hIn.length];
                    }
                    cache.doneMask = isTerminalBoundary;

                    PolicyValueNetwork.Output out = network.forward(buf.obs[t], hIn, cache);
                    caches.add(cache);

                    double newLogProb = PPOMath.jointLogProb(out,
                        buf.moveAction[t], buf.yawBucket[t], buf.pitchBucket[t],
                        buf.jumpAction[t], buf.sprintAction[t], buf.sneakAction[t], buf.attackAction[t]);

                    double ratio = Math.exp(newLogProb - buf.oldLogProb[t]);
                    double advantage = advantages[t];

                    double surrogate1 = ratio * advantage;
                    double surrogate2 = clip(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantage;
                    double policyObjective = Math.min(surrogate1, surrogate2);
                    sumPolicyLoss += -policyObjective;

                    double dObjective_dRatio = (advantage >= 0) 
                        ? ((ratio < 1 + clipEpsilon) ? advantage : 0.0)
                        : ((ratio > 1 - clipEpsilon) ? advantage : 0.0);

                    double dLoss_dNewLogProb = -dObjective_dRatio * ratio;
                    sumKl += (buf.oldLogProb[t] - newLogProb);

                    double vOld = buf.value[t], ret = returns[t], vUnclipped = out.value;
                    double vClipped = vOld + clip(vUnclipped - vOld, -clipEpsilon, clipEpsilon);
                    double lossUnclipped = (vUnclipped - ret) * (vUnclipped - ret);
                    double lossClipped = (vClipped - ret) * (vClipped - ret);
                    boolean useClipped = lossClipped > lossUnclipped;

                    sumValueLoss += valueCoef * Math.max(lossUnclipped, lossClipped);

                    double dValueLoss_dV = useClipped
                        ? 2.0 * (vClipped - ret) * (Math.abs(vUnclipped - vOld) <= clipEpsilon ? 1.0 : 0.0)
                        : 2.0 * (vUnclipped - ret);

                    double entropy = PPOMath.jointEntropy(out);
                    sumEntropy += entropy;

                    PolicyValueNetwork.HeadGrads hg = new PolicyValueNetwork.HeadGrads();
                    hg.dMoveLogits   = headGrad(out.moveLogits, buf.moveAction[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dYawLogits    = headGrad(out.yawLogits, buf.yawBucket[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dPitchLogits  = headGrad(out.pitchLogits, buf.pitchBucket[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dJumpLogits   = headGrad(out.jumpLogits, buf.jumpAction[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dSprintLogits = headGrad(out.sprintLogits, buf.sprintAction[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dSneakLogits  = headGrad(out.sneakLogits, buf.sneakAction[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dAttackLogits = headGrad(out.attackLogits, buf.attackAction[t], dLoss_dNewLogProb, entropyCoef);
                    hg.dValue = (float) (valueCoef * dValueLoss_dV);

                    grads.add(hg);
                    updateCount++;
                }

                network.backwardSegment(caches, grads);
                optimizer.clipGlobalNorm(maxGradNorm);
                stats.gradNorm = optimizer.globalGradNorm();
                optimizer.step();
            }
        }

        stats.meanPolicyLoss = sumPolicyLoss / Math.max(1, updateCount);
        stats.meanValueLoss = sumValueLoss / Math.max(1, updateCount);
        stats.meanEntropy = sumEntropy / Math.max(1, updateCount);
        stats.meanApproxKl = sumKl / Math.max(1, updateCount);
        return stats;
    }

    private static double[] headGrad(double[] logits, int action, double dLoss_dNewLogProb, double entropyCoef) {
        double[] probs = Categorical.softmax(logits);
        double H = Categorical.entropy(logits);
        double[] grad = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            grad[i] = dLoss_dNewLogProb * ((i == action ? 1.0 : 0.0) - probs[i]) - entropyCoef * (-probs[i] * (H + safeLog(probs[i])));
        }
        return grad;
    }

    private static double safeLog(double p) { return p > 1e-12 ? Math.log(p) : Math.log(1e-12); }
    private static double clip(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }
    private static float[] slice(float[] arr, int n) { float[] out = new float[n]; System.arraycopy(arr, 0, out, 0, n); return out; }
    private static boolean[] sliceBool(boolean[] arr, int n) { boolean[] out = new boolean[n]; System.arraycopy(arr, 0, out, 0, n); return out; }

    private static void normalizeInPlace(float[] a) {
        double mean = 0, var = 0;
        for (float v : a) mean += v; mean /= a.length;
        for (float v : a) var += (v - mean) * (v - mean); var /= a.length;
        double std = Math.sqrt(var) + 1e-8;
        for (int i = 0; i < a.length; i++) a[i] = (float) ((a[i] - mean) / std);
    }
}
