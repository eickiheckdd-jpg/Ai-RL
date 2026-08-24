package com.example.newgen6.rl;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Plain-Java PPO (clipped surrogate objective). No PyTorch, no autodiff -
 * gradients for the policy loss and the entropy bonus are derived by hand
 * and backpropagated through MLP.
 */
public class PPOAgent {

    private final MLP policyNet; // outputs logits, size = numActions
    private final MLP valueNet;  // outputs a single value estimate

    private final int numActions;
    private final Random rnd = new Random();

    private final List<Experience> buffer = new ArrayList<>();

    // Hyperparameters - tune these as training progresses
    public double gamma = 0.99;
    public double lambda = 0.95;
    public double clipEps = 0.2;
    public double entropyCoeff = 0.01;
    public double policyLR = 3e-4;
    public double valueLR = 1e-3;
    public int epochs = 4;
    public int updateEvery = 256; // env steps collected per PPO update

    public PPOAgent(int stateSize, int numActions, int[] hiddenLayers) {
        this.numActions = numActions;
        this.policyNet = new MLP(buildSizes(stateSize, hiddenLayers, numActions), 1234L);
        this.valueNet = new MLP(buildSizes(stateSize, hiddenLayers, 1), 5678L);
    }

    private int[] buildSizes(int in, int[] hidden, int out) {
        int[] s = new int[hidden.length + 2];
        s[0] = in;
        for (int i = 0; i < hidden.length; i++) s[i + 1] = hidden[i];
        s[s.length - 1] = out;
        return s;
    }

    public static class ActResult {
        public int action;
        public double logProb;
        public double value;
        public double[] probs;
    }

    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) max = Math.max(max, l);
        double sum = 0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }

    public ActResult act(double[] state) {
        double[] logits = policyNet.forward(state);
        double[] probs = softmax(logits);

        double sample = rnd.nextDouble();
        double cum = 0;
        int chosen = probs.length - 1;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (sample <= cum) { chosen = i; break; }
        }

        double value = valueNet.forward(state)[0];

        ActResult r = new ActResult();
        r.action = chosen;
        r.logProb = Math.log(Math.max(probs[chosen], 1e-8));
        r.value = value;
        r.probs = probs;
        return r;
    }

    public void remember(double[] state, int action, double logProb, double value, double reward, boolean done) {
        buffer.add(new Experience(state, action, logProb, value, reward, done));
        if (buffer.size() >= updateEvery) {
            update();
        }
    }

    private void computeGAE() {
        double nextValue = 0;
        double nextAdv = 0;
        for (int t = buffer.size() - 1; t >= 0; t--) {
            Experience e = buffer.get(t);
            double mask = e.done ? 0.0 : 1.0;
            double delta = e.reward + gamma * nextValue * mask - e.value;
            double adv = delta + gamma * lambda * mask * nextAdv;
            e.advantage = adv;
            e.returnTarget = adv + e.value;
            nextValue = e.value;
            nextAdv = adv;
        }
        double mean = 0;
        for (Experience e : buffer) mean += e.advantage;
        mean /= buffer.size();
        double var = 0;
        for (Experience e : buffer) var += (e.advantage - mean) * (e.advantage - mean);
        var /= Math.max(1, buffer.size());
        double std = Math.sqrt(var) + 1e-8;
        for (Experience e : buffer) e.advantage = (e.advantage - mean) / std;
    }

    public void update() {
        if (buffer.isEmpty()) return;
        computeGAE();

        List<Experience> batch = new ArrayList<>(buffer);
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(batch, rnd);
            for (Experience e : batch) {
                // ---- Policy (clipped surrogate) update ----
                double[] logits = policyNet.forward(e.state);
                double[] probs = softmax(logits);
                double newLogProb = Math.log(Math.max(probs[e.action], 1e-8));
                double ratio = Math.exp(newLogProb - e.logProb);

                double surr1 = ratio * e.advantage;
                double clippedRatio = Math.max(1 - clipEps, Math.min(1 + clipEps, ratio));
                double surr2 = clippedRatio * e.advantage;

                double dLoss_dNewLogProb;
                if (surr1 <= surr2) {
                    dLoss_dNewLogProb = -e.advantage * ratio;
                } else {
                    boolean clipIsActive = (ratio < 1 - clipEps || ratio > 1 + clipEps);
                    dLoss_dNewLogProb = clipIsActive ? 0.0 : -e.advantage * ratio;
                }

                double entropy = 0;
                for (double p : probs) if (p > 1e-8) entropy -= p * Math.log(p);

                double[] dLogits = new double[numActions];
                for (int j = 0; j < numActions; j++) {
                    double indicator = (j == e.action) ? 1.0 : 0.0;
                    double dPolicy = dLoss_dNewLogProb * (indicator - probs[j]);
                    double logPj = Math.log(Math.max(probs[j], 1e-8));
                    double dEntropy = entropyCoeff * probs[j] * (logPj + entropy);
                    dLogits[j] = dPolicy + dEntropy;
                }
                policyNet.backward(dLogits, policyLR);

                // ---- Value (MSE) update ----
                double predicted = valueNet.forward(e.state)[0];
                double dValue = predicted - e.returnTarget;
                valueNet.backward(new double[]{dValue}, valueLR);
            }
        }
        buffer.clear();
    }

    public void save(File file) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            policyNet.writeTo(out);
            valueNet.writeTo(out);
        }
    }

    public void load(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            MLP p = MLP.readFrom(in);
            MLP v = MLP.readFrom(in);
            this.policyNet.copyFrom(p);
            this.valueNet.copyFrom(v);
        }
    }
}
