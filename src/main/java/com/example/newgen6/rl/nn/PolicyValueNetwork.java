package com.example.newgen6.rl.nn;

import com.example.newgen6.rl.RLConstants;

import java.util.List;
import java.util.Random;

/**
 * Recurrent policy/value network:
 *
 * observation (229)
 *      ->
 * GRU (192)
 *      ->
 * trunk (128, ReLU)
 *      ->
 * independent categorical policy heads + scalar value
 */
public final class PolicyValueNetwork {

    public static final class Output {
        public double[] moveLogits;
        public double[] yawLogits;
        public double[] pitchLogits;
        public double[] jumpLogits;
        public double[] sprintLogits;
        public double[] sneakLogits;
        public double[] attackLogits;

        public double value;
        public float[] hiddenOut;
    }

    public static final class StepCache {
        public DenseLayer.Cache trunkCache;
        public DenseLayer.Cache moveCache;
        public DenseLayer.Cache yawCache;
        public DenseLayer.Cache pitchCache;
        public DenseLayer.Cache jumpCache;
        public DenseLayer.Cache sprintCache;
        public DenseLayer.Cache sneakCache;
        public DenseLayer.Cache attackCache;
        public DenseLayer.Cache valueCache;

        public GRUCell.Cache gruCache;

        public float[] obs;
        public float[] hiddenIn;
        public float[] trunkOut;
        public boolean doneMask;
    }

    public static final class HeadGrads {
        public double[] dMoveLogits;
        public double[] dYawLogits;
        public double[] dPitchLogits;
        public double[] dJumpLogits;
        public double[] dSprintLogits;
        public double[] dSneakLogits;
        public double[] dAttackLogits;
        public float dValue;
    }

    private final GRUCell gru;
    private final DenseLayer trunk;
    private final DenseLayer moveHead;
    private final DenseLayer yawHead;
    private final DenseLayer pitchHead;
    private final DenseLayer jumpHead;
    private final DenseLayer sprintHead;
    private final DenseLayer sneakHead;
    private final DenseLayer attackHead;
    private final DenseLayer valueHead;

    public PolicyValueNetwork(Random rng) {
        this(
                RLConstants.OBSERVATION_SIZE,
                RLConstants.GRU_HIDDEN_SIZE,
                rng
        );
    }

    public PolicyValueNetwork(int obsSize, int hiddenSize, Random rng) {
        if (obsSize != RLConstants.OBSERVATION_SIZE) {
            throw new IllegalArgumentException(
                    "Network observation size " + obsSize
                            + " != ABI size " + RLConstants.OBSERVATION_SIZE
            );
        }

        if (hiddenSize != RLConstants.GRU_HIDDEN_SIZE) {
            throw new IllegalArgumentException(
                    "Network hidden size " + hiddenSize
                            + " != ABI size " + RLConstants.GRU_HIDDEN_SIZE
            );
        }

        if (rng == null) {
            throw new IllegalArgumentException("rng cannot be null");
        }

        this.gru = new GRUCell(obsSize, hiddenSize, rng);
        this.trunk = new DenseLayer(
                hiddenSize,
                RLConstants.TRUNK_SIZE,
                DenseLayer.Activation.RELU,
                rng
        );

        this.moveHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.MOVE_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.yawHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.YAW_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.pitchHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.PITCH_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.jumpHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.JUMP_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.sprintHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.SPRINT_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.sneakHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.SNEAK_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.attackHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.ATTACK_ACTIONS,
                DenseLayer.Activation.NONE,
                rng
        );

        this.valueHead = new DenseLayer(
                RLConstants.TRUNK_SIZE,
                RLConstants.VALUE_OUTPUTS,
                DenseLayer.Activation.NONE,
                rng
        );
    }

    public Output forward(
            float[] obs,
            float[] hiddenIn,
            StepCache cache) {

        requireSize(obs, RLConstants.OBSERVATION_SIZE, "observation");
        requireSize(hiddenIn, RLConstants.GRU_HIDDEN_SIZE, "hiddenIn");

        if (cache != null) {
            cache.obs = obs;
            cache.hiddenIn = hiddenIn;
            cache.gruCache = new GRUCell.Cache();
            cache.trunkCache = new DenseLayer.Cache();
            cache.moveCache = new DenseLayer.Cache();
            cache.yawCache = new DenseLayer.Cache();
            cache.pitchCache = new DenseLayer.Cache();
            cache.jumpCache = new DenseLayer.Cache();
            cache.sprintCache = new DenseLayer.Cache();
            cache.sneakCache = new DenseLayer.Cache();
            cache.attackCache = new DenseLayer.Cache();
            cache.valueCache = new DenseLayer.Cache();
        }

        float[] hiddenOut = gru.forward(
                obs,
                hiddenIn,
                cache == null ? null : cache.gruCache
        );

        float[] trunkOut = trunk.forward(
                hiddenOut,
                cache == null ? null : cache.trunkCache
        );

        if (cache != null) {
            cache.trunkOut = trunkOut;
        }

        Output out = new Output();

        out.hiddenOut = hiddenOut;

        out.moveLogits = toDouble(moveHead.forward(
                trunkOut,
                cache == null ? null : cache.moveCache
        ));

        out.yawLogits = toDouble(yawHead.forward(
                trunkOut,
                cache == null ? null : cache.yawCache
        ));

        out.pitchLogits = toDouble(pitchHead.forward(
                trunkOut,
                cache == null ? null : cache.pitchCache
        ));

        out.jumpLogits = toDouble(jumpHead.forward(
                trunkOut,
                cache == null ? null : cache.jumpCache
        ));

        out.sprintLogits = toDouble(sprintHead.forward(
                trunkOut,
                cache == null ? null : cache.sprintCache
        ));

        out.sneakLogits = toDouble(sneakHead.forward(
                trunkOut,
                cache == null ? null : cache.sneakCache
        ));

        out.attackLogits = toDouble(attackHead.forward(
                trunkOut,
                cache == null ? null : cache.attackCache
        ));

        out.value = valueHead.forward(
                trunkOut,
                cache == null ? null : cache.valueCache
        )[0];

        if (!Double.isFinite(out.value)) {
            out.value = 0.0;
        }

        return out;
    }

    public void backwardSegment(
            List<StepCache> caches,
            List<HeadGrads> grads) {

        if (caches == null || grads == null) {
            throw new IllegalArgumentException("caches/grads cannot be null");
        }

        if (caches.size() != grads.size()) {
            throw new IllegalArgumentException(
                    "Cache/gradient sequence mismatch: "
                            + caches.size() + " != " + grads.size()
            );
        }

        float[] dHNext = new float[RLConstants.GRU_HIDDEN_SIZE];

        for (int t = caches.size() - 1; t >= 0; t--) {
            StepCache cache = caches.get(t);
            HeadGrads hg = grads.get(t);

            if (cache == null || hg == null) {
                throw new IllegalArgumentException(
                        "Null recurrent training element at t=" + t
                );
            }

            float[] dTrunk = new float[RLConstants.TRUNK_SIZE];

            accumulate(
                    dTrunk,
                    moveHead.backward(toFloat(hg.dMoveLogits), cache.moveCache)
            );
            accumulate(
                    dTrunk,
                    yawHead.backward(toFloat(hg.dYawLogits), cache.yawCache)
            );
            accumulate(
                    dTrunk,
                    pitchHead.backward(toFloat(hg.dPitchLogits), cache.pitchCache)
            );
            accumulate(
                    dTrunk,
                    jumpHead.backward(toFloat(hg.dJumpLogits), cache.jumpCache)
            );
            accumulate(
                    dTrunk,
                    sprintHead.backward(toFloat(hg.dSprintLogits), cache.sprintCache)
            );
            accumulate(
                    dTrunk,
                    sneakHead.backward(toFloat(hg.dSneakLogits), cache.sneakCache)
            );
            accumulate(
                    dTrunk,
                    attackHead.backward(toFloat(hg.dAttackLogits), cache.attackCache)
            );
            accumulate(
                    dTrunk,
                    valueHead.backward(
                            new float[]{ hg.dValue },
                            cache.valueCache
                    )
            );

            float[] dHiddenFromTrunk =
                    trunk.backward(dTrunk, cache.trunkCache);

            for (int i = 0; i < dHNext.length; i++) {
                dHNext[i] += dHiddenFromTrunk[i];
            }

            float[] dHPrev = gru.backward(
                    dHNext,
                    cache.obs,
                    cache.hiddenIn,
                    cache.gruCache
            );

            if (cache.doneMask) {
                for (int i = 0; i < dHPrev.length; i++) {
                    dHPrev[i] = 0.0f;
                }
            }

            dHNext = dHPrev;
        }
    }

    public void zeroGrad() {
        gru.zeroGrad();
        trunk.zeroGrad();
        moveHead.zeroGrad();
        yawHead.zeroGrad();
        pitchHead.zeroGrad();
        jumpHead.zeroGrad();
        sprintHead.zeroGrad();
        sneakHead.zeroGrad();
        attackHead.zeroGrad();
        valueHead.zeroGrad();
    }

    public void registerWith(AdamOptimizer optimizer) {
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }

        registerDense(optimizer, trunk);
        registerDense(optimizer, moveHead);
        registerDense(optimizer, yawHead);
        registerDense(optimizer, pitchHead);
        registerDense(optimizer, jumpHead);
        registerDense(optimizer, sprintHead);
        registerDense(optimizer, sneakHead);
        registerDense(optimizer, attackHead);
        registerDense(optimizer, valueHead);

        optimizer.register(gru.weightsZ(), gru.gradWeightsZ());
        optimizer.register(gru.recurrentWeightsZ(), gru.gradRecurrentWeightsZ());
        optimizer.register(gru.biasZ(), gru.gradBiasZ());

        optimizer.register(gru.weightsR(), gru.gradWeightsR());
        optimizer.register(gru.recurrentWeightsR(), gru.gradRecurrentWeightsR());
        optimizer.register(gru.biasR(), gru.gradBiasR());

        optimizer.register(gru.weightsN(), gru.gradWeightsN());
        optimizer.register(gru.recurrentWeightsN(), gru.gradRecurrentWeightsN());
        optimizer.register(gru.biasN(), gru.gradBiasN());
    }

    private static void registerDense(
            AdamOptimizer optimizer,
            DenseLayer layer) {

        optimizer.register(layer.weights(), layer.gradWeights());
        optimizer.register(layer.bias(), layer.gradBias());
    }

    public GRUCell gru() {
        return gru;
    }

    public DenseLayer trunk() {
        return trunk;
    }

    public DenseLayer moveHead() {
        return moveHead;
    }

    public DenseLayer yawHead() {
        return yawHead;
    }

    public DenseLayer pitchHead() {
        return pitchHead;
    }

    public DenseLayer jumpHead() {
        return jumpHead;
    }

    public DenseLayer sprintHead() {
        return sprintHead;
    }

    public DenseLayer sneakHead() {
        return sneakHead;
    }

    public DenseLayer attackHead() {
        return attackHead;
    }

    public DenseLayer valueHead() {
        return valueHead;
    }

    private static void requireSize(
            float[] array,
            int expected,
            String name) {

        if (array == null || array.length != expected) {
            throw new IllegalArgumentException(
                    name + " size mismatch: got "
                            + (array == null ? "null" : array.length)
                            + ", expected " + expected
            );
        }
    }

    private static double[] toDouble(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    private static float[] toFloat(double[] values) {
        if (values == null) {
            throw new IllegalArgumentException("gradient cannot be null");
        }

        float[] result = new float[values.length];

        for (int i = 0; i < values.length; i++) {
            result[i] = Double.isFinite(values[i])
                    ? (float) values[i]
                    : 0.0f;
        }

        return result;
    }

    private static void accumulate(float[] target, float[] source) {
        if (target.length != source.length) {
            throw new IllegalArgumentException(
                    "Gradient dimension mismatch: "
                            + target.length + " != " + source.length
            );
        }

        for (int i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }
}