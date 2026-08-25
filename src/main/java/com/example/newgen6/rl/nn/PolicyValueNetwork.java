package com.example.newgen6.rl.nn;

import java.util.Random;

public class PolicyValueNetwork {
    public static final int HIDDEN_SIZE = 128;

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

    public static class Cache {
        public GRUCell.Cache gruCache = new GRUCell.Cache();
        public DenseLayer.Cache denseCache = new DenseLayer.Cache();
        public DenseLayer.Cache moveCache = new DenseLayer.Cache();
        public DenseLayer.Cache yawCache = new DenseLayer.Cache();
        public DenseLayer.Cache pitchCache = new DenseLayer.Cache();
        public DenseLayer.Cache jumpCache = new DenseLayer.Cache();
        public DenseLayer.Cache sprintCache = new DenseLayer.Cache();
        public DenseLayer.Cache sneakCache = new DenseLayer.Cache();
        public DenseLayer.Cache attackCache = new DenseLayer.Cache();
        public DenseLayer.Cache valueCache = new DenseLayer.Cache();
    }

    public static class Output {
        public float[] h;
        public float value;
        public float[] moveLogits;
        public float[] yawLogits;
        public float[] pitchLogits;
        public float[] jumpLogits;
        public float[] sprintLogits;
        public float[] sneakLogits;
        public float[] attackLogits;
    }

    public PolicyValueNetwork(int inputDim, int hiddenSize, Random rng) {
        this.gru = new GRUCell(inputDim, hiddenSize, rng);
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

    public float[] initialHiddenState() {
        return new float[HIDDEN_SIZE];
    }

    public void registerWith(AdamOptimizer optimizer) {
        gru.registerWith(optimizer);
        trunk.registerWith(optimizer);
        moveHead.registerWith(optimizer);
        yawHead.registerWith(optimizer);
        pitchHead.registerWith(optimizer);
        jumpHead.registerWith(optimizer);
        sprintHead.registerWith(optimizer);
        sneakHead.registerWith(optimizer);
        attackHead.registerWith(optimizer);
        valueHead.registerWith(optimizer);
    }
}
