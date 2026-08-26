package com.example.newgen6.rl;

import java.util.Random;

/**
 * Pure-Java policy for 229F × 200T context.
 *
 * Temporal path (mobile-friendly, not a full 200-step transformer):
 *   - encode last TEMPORAL_FRAMES observations → FRAME_ENC each
 *   - encode history mean → FRAME_ENC
 *   - concat → trunk MLP → TRUNK_SIZE
 *   - heads: move(9), jump/sprint/attack/sneak logits, yaw(19), pitch(17), value
 *
 * Trainable from random init. No external NN runtime.
 */
public final class PolicyNetwork {
    private final Random rng;
    private final DenseLayer frameEnc;
    private final DenseLayer trunk1;
    private final DenseLayer trunk2;
    private final DenseLayer moveHead;
    private final DenseLayer binaryHead; // 4 logits: jump,sprint,attack,sneak
    private final DenseLayer yawHead;
    private final DenseLayer pitchHead;
    private final DenseLayer valueHead;

    private final float[] frameBuf = new float[RLConstants.FRAME_ENC];
    private final float[] temporalIn;
    private final float[] h1 = new float[RLConstants.HIDDEN];
    private final float[] trunk = new float[RLConstants.TRUNK_SIZE];
    private final float[] moveLogits = new float[RLConstants.MOVE_ACTIONS];
    private final float[] binLogits = new float[4];
    private final float[] yawLogits = new float[RLConstants.YAW_BUCKETS];
    private final float[] pitchLogits = new float[RLConstants.PITCH_BUCKETS];
    private final float[] valueOut = new float[1];

    private final float[] obsScratch = new float[RLConstants.OBSERVATION_SIZE];
    private final float[] meanScratch = new float[RLConstants.OBSERVATION_SIZE];

    private int adamT = 0;

    public PolicyNetwork(long seed) {
        this.rng = new Random(seed);
        int temporalInSize = RLConstants.FRAME_ENC * (RLConstants.TEMPORAL_FRAMES + 1);
        this.temporalIn = new float[temporalInSize];

        this.frameEnc = new DenseLayer(RLConstants.OBSERVATION_SIZE, RLConstants.FRAME_ENC, true, rng);
        this.trunk1 = new DenseLayer(temporalInSize, RLConstants.HIDDEN, true, rng);
        this.trunk2 = new DenseLayer(RLConstants.HIDDEN, RLConstants.TRUNK_SIZE, true, rng);
        this.moveHead = new DenseLayer(RLConstants.TRUNK_SIZE, RLConstants.MOVE_ACTIONS, false, rng);
        this.binaryHead = new DenseLayer(RLConstants.TRUNK_SIZE, 4, false, rng);
        this.yawHead = new DenseLayer(RLConstants.TRUNK_SIZE, RLConstants.YAW_BUCKETS, false, rng);
        this.pitchHead = new DenseLayer(RLConstants.TRUNK_SIZE, RLConstants.PITCH_BUCKETS, false, rng);
        this.valueHead = new DenseLayer(RLConstants.TRUNK_SIZE, 1, false, rng);
    }

    private void buildTemporalInput(ContextBuffer ctx) {
        int pos = 0;
        for (int age = 0; age < RLConstants.TEMPORAL_FRAMES; age++) {
            ctx.copyAge(age, obsScratch);
            frameEnc.forward(obsScratch, frameBuf);
            System.arraycopy(frameBuf, 0, temporalIn, pos, RLConstants.FRAME_ENC);
            pos += RLConstants.FRAME_ENC;
        }
        ctx.mean(meanScratch);
        frameEnc.forward(meanScratch, frameBuf);
        System.arraycopy(frameBuf, 0, temporalIn, pos, RLConstants.FRAME_ENC);
    }

    private void forwardTrunk() {
        trunk1.forward(temporalIn, h1);
        trunk2.forward(h1, trunk);
        moveHead.forward(trunk, moveLogits);
        binaryHead.forward(trunk, binLogits);
        yawHead.forward(trunk, yawLogits);
        pitchHead.forward(trunk, pitchLogits);
        valueHead.forward(trunk, valueOut);
    }

    public ActionSample act(ContextBuffer ctx, boolean deterministic) {
        buildTemporalInput(ctx);
        forwardTrunk();

        float[] moveP = moveLogits.clone();
        MathUtil.softmaxInPlace(moveP, 0, RLConstants.MOVE_ACTIONS);
        float[] yawP = yawLogits.clone();
        MathUtil.softmaxInPlace(yawP, 0, RLConstants.YAW_BUCKETS);
        float[] pitchP = pitchLogits.clone();
        MathUtil.softmaxInPlace(pitchP, 0, RLConstants.PITCH_BUCKETS);

        ActionSample a = new ActionSample();
        if (deterministic) {
            a.move = argmax(moveP, RLConstants.MOVE_ACTIONS);
            a.yawBucket = argmax(yawP, RLConstants.YAW_BUCKETS);
            a.pitchBucket = argmax(pitchP, RLConstants.PITCH_BUCKETS);
            a.jump = MathUtil.sigmoid(binLogits[0]) > 0.5f;
            a.sprint = MathUtil.sigmoid(binLogits[1]) > 0.5f;
            a.attack = MathUtil.sigmoid(binLogits[2]) > 0.5f;
            a.sneak = MathUtil.sigmoid(binLogits[3]) > 0.5f;
        } else {
            a.move = MathUtil.sampleCategorical(moveP, 0, RLConstants.MOVE_ACTIONS, rng);
            a.yawBucket = MathUtil.sampleCategorical(yawP, 0, RLConstants.YAW_BUCKETS, rng);
            a.pitchBucket = MathUtil.sampleCategorical(pitchP, 0, RLConstants.PITCH_BUCKETS, rng);
            a.jump = rng.nextFloat() < MathUtil.sigmoid(binLogits[0]);
            a.sprint = rng.nextFloat() < MathUtil.sigmoid(binLogits[1]);
            a.attack = rng.nextFloat() < MathUtil.sigmoid(binLogits[2]);
            a.sneak = rng.nextFloat() < MathUtil.sigmoid(binLogits[3]);
        }

        a.logProb = jointLogProb(moveP, yawP, pitchP, a);
        a.entropy = entropyCat(moveP, RLConstants.MOVE_ACTIONS)
                + entropyCat(yawP, RLConstants.YAW_BUCKETS)
                + entropyCat(pitchP, RLConstants.PITCH_BUCKETS)
                + bernEntropy(MathUtil.sigmoid(binLogits[0]))
                + bernEntropy(MathUtil.sigmoid(binLogits[1]))
                + bernEntropy(MathUtil.sigmoid(binLogits[2]))
                + bernEntropy(MathUtil.sigmoid(binLogits[3]));
        a.value = valueOut[0];
        a.applyBuckets();
        return a;
    }

    public float evaluateValue(ContextBuffer ctx) {
        buildTemporalInput(ctx);
        forwardTrunk();
        return valueOut[0];
    }

    /** PPO loss step on one transition (stores grads). */
    public void accumulatePpoGrads(
            ContextBuffer ctx,
            ActionSample taken,
            float advantage,
            float ret,
            float oldLogProb,
            float clip) {

        buildTemporalInput(ctx);
        forwardTrunk();

        float[] moveP = moveLogits.clone();
        MathUtil.softmaxInPlace(moveP, 0, RLConstants.MOVE_ACTIONS);
        float[] yawP = yawLogits.clone();
        MathUtil.softmaxInPlace(yawP, 0, RLConstants.YAW_BUCKETS);
        float[] pitchP = pitchLogits.clone();
        MathUtil.softmaxInPlace(pitchP, 0, RLConstants.PITCH_BUCKETS);

        float newLogProb = jointLogProb(moveP, yawP, pitchP, taken);
        float ratio = (float) Math.exp(newLogProb - oldLogProb);
        float unclipped = ratio * advantage;
        float clipped = MathUtil.clamp(ratio, 1f - clip, 1f + clip) * advantage;
        float policyLoss = -Math.min(unclipped, clipped);

        // Simple surrogate gradient via logit nudging (REINFORCE-style on discrete heads)
        float scale = policyLoss; // magnitude; sign handled per-action below
        float advSign = advantage >= 0 ? 1f : -1f;
        float ratioFactor = (ratio > 1f + clip && advantage > 0) || (ratio < 1f - clip && advantage < 0)
                ? 0f : 1f;

        float[] dMove = new float[RLConstants.MOVE_ACTIONS];
        float[] dYaw = new float[RLConstants.YAW_BUCKETS];
        float[] dPitch = new float[RLConstants.PITCH_BUCKETS];
        float[] dBin = new float[4];

        // Gradient of log π(a) w.r.t. logits ≈ 1_{a} - p
        for (int i = 0; i < RLConstants.MOVE_ACTIONS; i++) {
            dMove[i] = ((i == taken.move ? 1f : 0f) - moveP[i]) * advSign * ratioFactor;
        }
        for (int i = 0; i < RLConstants.YAW_BUCKETS; i++) {
            dYaw[i] = ((i == taken.yawBucket ? 1f : 0f) - yawP[i]) * advSign * ratioFactor;
        }
        for (int i = 0; i < RLConstants.PITCH_BUCKETS; i++) {
            dPitch[i] = ((i == taken.pitchBucket ? 1f : 0f) - pitchP[i]) * advSign * ratioFactor;
        }

        float p0 = MathUtil.sigmoid(binLogits[0]);
        float p1 = MathUtil.sigmoid(binLogits[1]);
        float p2 = MathUtil.sigmoid(binLogits[2]);
        float p3 = MathUtil.sigmoid(binLogits[3]);
        dBin[0] = ((taken.jump ? 1f : 0f) - p0) * advSign * ratioFactor;
        dBin[1] = ((taken.sprint ? 1f : 0f) - p1) * advSign * ratioFactor;
        dBin[2] = ((taken.attack ? 1f : 0f) - p2) * advSign * ratioFactor;
        dBin[3] = ((taken.sneak ? 1f : 0f) - p3) * advSign * ratioFactor;

        // Entropy bonus gradient (encourage exploration)
        float entCoef = RLConstants.ENTROPY_COEF;
        for (int i = 0; i < RLConstants.MOVE_ACTIONS; i++) {
            dMove[i] += entCoef * (-(float) Math.log(Math.max(moveP[i], 1e-8f)) - 1f) * moveP[i];
        }

        float[] dValue = new float[] { RLConstants.VALUE_COEF * 2f * (valueOut[0] - ret) };

        // Backprop heads → trunk → temporal (frameEnc grads only from last forward mean path approx)
        float[] dTrunk = new float[RLConstants.TRUNK_SIZE];
        moveHead.backward(dMove, dTrunk);
        float[] dTrunkB = new float[RLConstants.TRUNK_SIZE];
        binaryHead.backward(dBin, dTrunkB);
        MathUtil.axpy(dTrunk, dTrunkB, 1f, RLConstants.TRUNK_SIZE);
        float[] dTrunkY = new float[RLConstants.TRUNK_SIZE];
        yawHead.backward(dYaw, dTrunkY);
        MathUtil.axpy(dTrunk, dTrunkY, 1f, RLConstants.TRUNK_SIZE);
        float[] dTrunkP = new float[RLConstants.TRUNK_SIZE];
        pitchHead.backward(dPitch, dTrunkP);
        MathUtil.axpy(dTrunk, dTrunkP, 1f, RLConstants.TRUNK_SIZE);
        float[] dTrunkV = new float[RLConstants.TRUNK_SIZE];
        valueHead.backward(dValue, dTrunkV);
        MathUtil.axpy(dTrunk, dTrunkV, 1f, RLConstants.TRUNK_SIZE);

        float[] dH1 = new float[RLConstants.HIDDEN];
        trunk2.backward(dTrunk, dH1);
        float[] dTemporal = new float[temporalIn.length];
        trunk1.backward(dH1, dTemporal);

        // Backprop frame encoder on the mean path (last FRAME_ENC slice) + age-0 path
        float[] dFrame = new float[RLConstants.FRAME_ENC];
        int meanOff = RLConstants.FRAME_ENC * RLConstants.TEMPORAL_FRAMES;
        System.arraycopy(dTemporal, meanOff, dFrame, 0, RLConstants.FRAME_ENC);
        float[] dObs = new float[RLConstants.OBSERVATION_SIZE];
        // Re-run mean encoding path for correct lastIn
        ctx.mean(meanScratch);
        frameEnc.forward(meanScratch, frameBuf);
        frameEnc.backward(dFrame, dObs);

        float[] dFrame0 = new float[RLConstants.FRAME_ENC];
        System.arraycopy(dTemporal, 0, dFrame0, 0, RLConstants.FRAME_ENC);
        ctx.copyAge(0, obsScratch);
        frameEnc.forward(obsScratch, frameBuf);
        float[] dObs0 = new float[RLConstants.OBSERVATION_SIZE];
        frameEnc.backward(dFrame0, dObs0);
    }

    public void zeroGrads() {
        frameEnc.zeroGrads();
        trunk1.zeroGrads();
        trunk2.zeroGrads();
        moveHead.zeroGrads();
        binaryHead.zeroGrads();
        yawHead.zeroGrads();
        pitchHead.zeroGrads();
        valueHead.zeroGrads();
    }

    public void adamStep(float lr) {
        adamT++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        frameEnc.adamStep(lr, b1, b2, eps, adamT);
        trunk1.adamStep(lr, b1, b2, eps, adamT);
        trunk2.adamStep(lr, b1, b2, eps, adamT);
        moveHead.adamStep(lr, b1, b2, eps, adamT);
        binaryHead.adamStep(lr, b1, b2, eps, adamT);
        yawHead.adamStep(lr, b1, b2, eps, adamT);
        pitchHead.adamStep(lr, b1, b2, eps, adamT);
        valueHead.adamStep(lr, b1, b2, eps, adamT);
    }

    private static int argmax(float[] p, int n) {
        int best = 0;
        float v = p[0];
        for (int i = 1; i < n; i++) {
            if (p[i] > v) {
                v = p[i];
                best = i;
            }
        }
        return best;
    }

    private float jointLogProb(float[] moveP, float[] yawP, float[] pitchP, ActionSample a) {
        float lp = MathUtil.logProbCategorical(moveP, 0, a.move);
        lp += MathUtil.logProbCategorical(yawP, 0, a.yawBucket);
        lp += MathUtil.logProbCategorical(pitchP, 0, a.pitchBucket);
        float p0 = MathUtil.sigmoid(binLogits[0]);
        float p1 = MathUtil.sigmoid(binLogits[1]);
        float p2 = MathUtil.sigmoid(binLogits[2]);
        float p3 = MathUtil.sigmoid(binLogits[3]);
        lp += (float) Math.log(Math.max(a.jump ? p0 : 1f - p0, 1e-8f));
        lp += (float) Math.log(Math.max(a.sprint ? p1 : 1f - p1, 1e-8f));
        lp += (float) Math.log(Math.max(a.attack ? p2 : 1f - p2, 1e-8f));
        lp += (float) Math.log(Math.max(a.sneak ? p3 : 1f - p3, 1e-8f));
        return lp;
    }

    private static float entropyCat(float[] p, int n) {
        float h = 0f;
        for (int i = 0; i < n; i++) {
            float x = Math.max(p[i], 1e-8f);
            h -= x * (float) Math.log(x);
        }
        return h;
    }

    private static float bernEntropy(float p) {
        p = MathUtil.clamp(p, 1e-8f, 1f - 1e-8f);
        return (float) (-p * Math.log(p) - (1 - p) * Math.log(1 - p));
    }

    public float[] lastMoveProbs() {
        float[] p = moveLogits.clone();
        MathUtil.softmaxInPlace(p, 0, RLConstants.MOVE_ACTIONS);
        return p;
    }

    public float[] lastYawProbs() {
        float[] p = yawLogits.clone();
        MathUtil.softmaxInPlace(p, 0, RLConstants.YAW_BUCKETS);
        return p;
    }

    public float[] lastPitchProbs() {
        float[] p = pitchLogits.clone();
        MathUtil.softmaxInPlace(p, 0, RLConstants.PITCH_BUCKETS);
        return p;
    }
}
