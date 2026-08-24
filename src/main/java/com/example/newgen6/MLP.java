package com.example.newgen6.rl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * A small fully-connected feedforward network with hand-written backprop and
 * a per-sample Adam update. No autodiff library, no PyTorch, pure Java.
 *
 * Hidden layers use tanh. The output layer is linear (logits for the policy
 * net, a raw scalar for the value net) - softmax/interpretation happens in
 * PPOAgent, not here.
 */
public class MLP {

    private final int[] sizes;
    private final double[][][] weights; // [layer][outIdx][inIdx]
    private final double[][] biases;    // [layer][outIdx]

    // Adam moment estimates (same shape as weights/biases)
    private final double[][][] mW, vW;
    private final double[][] mB, vB;
    private int adamStep = 0;

    private static final double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;

    private transient double[][] lastZ; // pre-activations from the last forward()
    private transient double[][] lastA; // activations, lastA[0] = input

    public MLP(int[] sizes, long seed) {
        this.sizes = sizes;
        int L = sizes.length - 1;
        weights = new double[L][][];
        biases = new double[L][];
        mW = new double[L][][];
        vW = new double[L][][];
        mB = new double[L][];
        vB = new double[L][];
        Random rnd = new Random(seed);
        for (int l = 0; l < L; l++) {
            int in = sizes[l], out = sizes[l + 1];
            weights[l] = new double[out][in];
            mW[l] = new double[out][in];
            vW[l] = new double[out][in];
            biases[l] = new double[out];
            mB[l] = new double[out];
            vB[l] = new double[out];
            double scale = Math.sqrt(2.0 / in);
            for (int o = 0; o < out; o++) {
                for (int i = 0; i < in; i++) {
                    weights[l][o][i] = rnd.nextGaussian() * scale;
                }
            }
        }
    }

    public double[] forward(double[] input) {
        int L = weights.length;
        lastZ = new double[L][];
        lastA = new double[L + 1][];
        lastA[0] = input;
        double[] a = input;
        for (int l = 0; l < L; l++) {
            int out = sizes[l + 1];
            double[] z = new double[out];
            for (int o = 0; o < out; o++) {
                double sum = biases[l][o];
                double[] wRow = weights[l][o];
                for (int i = 0; i < a.length; i++) sum += wRow[i] * a[i];
                z[o] = sum;
            }
            lastZ[l] = z;
            boolean isOutputLayer = (l == L - 1);
            double[] act = new double[out];
            for (int o = 0; o < out; o++) {
                act[o] = isOutputLayer ? z[o] : Math.tanh(z[o]);
            }
            lastA[l + 1] = act;
            a = act;
        }
        return a;
    }

    /**
     * Backprop given dLoss/dOutput for the sample just run through forward(),
     * then applies an Adam step immediately (per-sample, not batch-averaged -
     * simpler to implement correctly, a bit noisier, works fine at this scale).
     */
    public void backward(double[] dOutput, double learningRate) {
        int L = weights.length;
        double[] delta = dOutput.clone();
        adamStep++;
        for (int l = L - 1; l >= 0; l--) {
            boolean isOutputLayer = (l == L - 1);
            if (!isOutputLayer) {
                double[] z = lastZ[l];
                for (int o = 0; o < delta.length; o++) {
                    double t = Math.tanh(z[o]);
                    delta[o] *= (1 - t * t);
                }
            }
            double[] aPrev = lastA[l];
            double[][] wLayer = weights[l];

            // Gradient w.r.t. previous layer's activations, using PRE-update weights
            double[] gradPrev = new double[aPrev.length];
            for (int i = 0; i < aPrev.length; i++) {
                double sum = 0;
                for (int o = 0; o < delta.length; o++) sum += wLayer[o][i] * delta[o];
                gradPrev[i] = sum;
            }

            // Adam update for this layer's weights and biases
            for (int o = 0; o < delta.length; o++) {
                double d = delta[o];

                double gB = d;
                mB[l][o] = BETA1 * mB[l][o] + (1 - BETA1) * gB;
                vB[l][o] = BETA2 * vB[l][o] + (1 - BETA2) * gB * gB;
                double mHatB = mB[l][o] / (1 - Math.pow(BETA1, adamStep));
                double vHatB = vB[l][o] / (1 - Math.pow(BETA2, adamStep));
                biases[l][o] -= learningRate * mHatB / (Math.sqrt(vHatB) + EPS);

                double[] wRow = wLayer[o];
                for (int i = 0; i < aPrev.length; i++) {
                    double grad = d * aPrev[i];
                    mW[l][o][i] = BETA1 * mW[l][o][i] + (1 - BETA1) * grad;
                    vW[l][o][i] = BETA2 * vW[l][o][i] + (1 - BETA2) * grad * grad;
                    double mHat = mW[l][o][i] / (1 - Math.pow(BETA1, adamStep));
                    double vHat = vW[l][o][i] / (1 - Math.pow(BETA2, adamStep));
                    wRow[i] -= learningRate * mHat / (Math.sqrt(vHat) + EPS);
                }
            }
            delta = gradPrev;
        }
    }

    public void copyFrom(MLP other) {
        for (int l = 0; l < weights.length; l++) {
            for (int o = 0; o < weights[l].length; o++) {
                System.arraycopy(other.weights[l][o], 0, weights[l][o], 0, weights[l][o].length);
            }
            System.arraycopy(other.biases[l], 0, biases[l], 0, biases[l].length);
        }
    }

    public void writeTo(DataOutputStream out) throws IOException {
        out.writeInt(sizes.length);
        for (int s : sizes) out.writeInt(s);
        for (double[][] layer : weights)
            for (double[] row : layer)
                for (double w : row) out.writeDouble(w);
        for (double[] layer : biases)
            for (double b : layer) out.writeDouble(b);
    }

    public static MLP readFrom(DataInputStream in) throws IOException {
        int n = in.readInt();
        int[] sizes = new int[n];
        for (int i = 0; i < n; i++) sizes[i] = in.readInt();
        MLP net = new MLP(sizes, 0L);
        for (double[][] layer : net.weights)
            for (double[] row : layer)
                for (int i = 0; i < row.length; i++) row[i] = in.readDouble();
        for (double[] layer : net.biases)
            for (int i = 0; i < layer.length; i++) layer[i] = in.readDouble();
        return net;
    }
}
