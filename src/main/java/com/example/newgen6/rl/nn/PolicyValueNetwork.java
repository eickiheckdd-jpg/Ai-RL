package com.example.newgen6.rl.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class PolicyValueNetwork {

    public static final class Output {
        public double[] moveLogits, yawLogits, pitchLogits, jumpLogits, sprintLogits, sneakLogits, attackLogits;
        public double value;
        public float[] hiddenOut;
    }

    public static final class StepCache {
        public float[] obs;
        public float[] hiddenIn;
        public GRUCell.Cache gruCache;
        public DenseLayer.Cache denseCache, moveCache, yawCache, pitchCache, jumpCache, sprintCache, sneakCache, attackCache, valueCache;
        public float[] trunkOut;
        public boolean doneMask; 
    }

    public static final class HeadGrads {
        public double[] dMoveLogits, dYawLogits, dPitchLogits, dJumpLogits, dSprintLogits, dSneakLogits, dAttackLogits;
        public float dValue;
    }

    private final GRUCell gru;
    private final DenseLayer trunk, moveHead, yawHead, pitchHead, jumpHead, sprintHead, sneakHead, attackHead, valueHead;

    public PolicyValueNetwork(int obsSize, int hiddenSize, Random rng) {
        this.gru = new GRUCell(obsSize, hiddenSize, rng);
        this.trunk = new DenseLayer(hiddenSize, 128, DenseLayer.Activation.RELU, rng);
        this.moveHead = new DenseLayer(128, 9, DenseLayer.Activation.NONE, rng);
        this.yawHead = new DenseLayer(128, 19, DenseLayer.Activation.NONE, rng);
        this.pitchHead = new DenseLayer(128, 17, DenseLayer.Activation.NONE, rng);
        this.jumpHead = new DenseLayer(128, 2, DenseLayer.Activation.NONE, rng);
        this.sprintHead = new DenseLayer(128, 2, DenseLayer.Activation.NONE, rng);
        this.sneakHead = new DenseLayer(128, 2, DenseLayer.Activation.NONE, rng);
        this.attackHead = new DenseLayer(128, 2, DenseLayer.Activation.NONE, rng);
        this.valueHead = new DenseLayer(128, 1, DenseLayer.Activation.NONE, rng);
    }

    public Output forward(float[] obs, float[] hiddenIn, StepCache cache) {
        if (cache != null) {
            cache.obs = obs; cache.hiddenIn = hiddenIn;
            cache.gruCache = new GRUCell.Cache(); cache.denseCache = new DenseLayer.Cache();
            cache.moveCache = new DenseLayer.Cache(); cache.yawCache = new DenseLayer.Cache();
            cache.pitchCache = new DenseLayer.Cache(); cache.jumpCache = new DenseLayer.Cache();
            cache.sprintCache = new DenseLayer.Cache(); cache.sneakCache = new DenseLayer.Cache();
            cache.attackCache = new DenseLayer.Cache(); cache.valueCache = new DenseLayer.Cache();
        }

        float[] hOut = gru.forward(obs, hiddenIn, cache != null ? cache.gruCache : null);
        float[] trunkOut = trunk.forward(hOut, cache != null ? cache.denseCache : null);
        if (cache != null) cache.trunkOut = trunkOut;

        Output out = new Output();
        out.hiddenOut = hOut;
        out.moveLogits = toDouble(moveHead.forward(trunkOut, cache != null ? cache.moveCache : null));
        out.yawLogits = toDouble(yawHead.forward(trunkOut, cache != null ? cache.yawCache : null));
        out.pitchLogits = toDouble(pitchHead.forward(trunkOut, cache != null ? cache.pitchCache : null));
        out.jumpLogits = toDouble(jumpHead.forward(trunkOut, cache != null ? cache.jumpCache : null));
        out.sprintLogits = toDouble(sprintHead.forward(trunkOut, cache != null ? cache.sprintCache : null));
        out.sneakLogits = toDouble(sneakHead.forward(trunkOut, cache != null ? cache.sneakCache : null));
        out.attackLogits = toDouble(attackHead.forward(trunkOut, cache != null ? cache.attackCache : null));
        out.value = valueHead.forward(trunkOut, cache != null ? cache.valueCache : null)[0];

        return out;
    }

    public void backwardSegment(List<StepCache> caches, List<HeadGrads> grads) {
        float[] dHNext = new float[gru.getHiddenSize()];

        for (int t = caches.size() - 1; t >= 0; t--) {
            StepCache cache = caches.get(t);
            HeadGrads hg = grads.get(t);

            float[] dTrunk = new float[128];
            accumulate(dTrunk, moveHead.backward(toFloat(hg.dMoveLogits), cache.moveCache));
            accumulate(dTrunk, yawHead.backward(toFloat(hg.dYawLogits), cache.yawCache));
            accumulate(dTrunk, pitchHead.backward(toFloat(hg.dPitchLogits), cache.pitchCache));
            accumulate(dTrunk, jumpHead.backward(toFloat(hg.dJumpLogits), cache.jumpCache));
            accumulate(dTrunk, sprintHead.backward(toFloat(hg.dSprintLogits), cache.sprintCache));
            accumulate(dTrunk, sneakHead.backward(toFloat(hg.dSneakLogits), cache.sneakCache));
            accumulate(dTrunk, attackHead.backward(toFloat(hg.dAttackLogits), cache.attackCache));
            accumulate(dTrunk, valueHead.backward(new float[]{ hg.dValue }, cache.valueCache));

            float[] dH_from_trunk = trunk.backward(dTrunk, cache.denseCache);
            for (int i = 0; i < dHNext.length; i++) {
                dHNext[i] += dH_from_trunk[i];
            }

            float[] dHPrev = gru.backward(dHNext, cache.obs, cache.hiddenIn, cache.gruCache);

            // Mask hidden gradient flow across episode boundaries
            if (cache.doneMask) {
                for (int i = 0; i < dHPrev.length; i++) dHPrev[i] = 0.0f;
            }
            dHNext = dHPrev;
        }
    }

    public void zeroGrad() {
        gru.zeroGrad(); trunk.zeroGrad();
        moveHead.zeroGrad(); yawHead.zeroGrad(); pitchHead.zeroGrad();
        jumpHead.zeroGrad(); sprintHead.zeroGrad(); sneakHead.zeroGrad();
        attackHead.zeroGrad(); valueHead.zeroGrad();
    }

    private static double[] toDouble(float[] in) {
        double[] out = new double[in.length];
        for (int i = 0; i < in.length; i++) out[i] = in[i];
        return out;
    }

    private static float[] toFloat(double[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) out[i] = (float) in[i];
        return out;
    }

    private static void accumulate(float[] target, float[] src) {
        for (int i = 0; i < target.length; i++) target[i] += src[i];
    }
}
