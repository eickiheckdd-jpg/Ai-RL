package com.example.newgen6.rl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public final class Mlp {
    public final int[] sizes;
    public final int layers;

    public final float[][][] weights;
    public final float[][] biases;

    public final float[][][] gradW;
    public final float[][] gradB;

    public final float[][] acts;
    public final float[][] pre;
    private final float[][] deltas;

    private final float[][][] mW;
    private final float[][][] vW;
    private final float[][] mB;
    private final float[][] vB;

    public float learningRate;

    private float beta1Pow = 1.0f;
    private float beta2Pow = 1.0f;

    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float EPS = 1e-6f;

    public Mlp(int[] sizes, Random rng, float learningRate) {
        this.sizes = sizes.clone();
        this.layers = this.sizes.length;
        this.learningRate = learningRate;

        weights = new float[layers - 1][][];
        biases = new float[layers - 1][];
        gradW = new float[layers - 1][][];
        gradB = new float[layers - 1][];

        mW = new float[layers - 1][][];
        vW = new float[layers - 1][][];
        mB = new float[layers - 1][];
        vB = new float[layers - 1][];

        acts = new float[layers][];
        pre = new float[layers][];
        deltas = new float[layers][];

        for (int l = 0; l < layers; l++) {
            acts[l] = new float[this.sizes[l]];
            deltas[l] = new float[this.sizes[l]];
            if (l > 0) {
                pre[l] = new float[this.sizes[l]];
            }
        }

        for (int l = 0; l < layers - 1; l++) {
            int in = this.sizes[l];
            int out = this.sizes[l + 1];

            weights[l] = new float[out][in];
            biases[l] = new float[out];

            gradW[l] = new float[out][in];
            gradB[l] = new float[out];

            mW[l] = new float[out][in];
            vW[l] = new float[out][in];

            mB[l] = new float[out];
            vB[l] = new float[out];

            float limit = (float) Math.sqrt(6.0 / (in + out));
            for (int o = 0; o < out; o++) {
                for (int i = 0; i < in; i++) {
                    weights[l][o][i] = (rng.nextFloat() * 2.0f - 1.0f) * limit;
                }
                biases[l][o] = 0.0f;
            }
        }
    }

    public Mlp(Mlp other) {
        this(other.sizes, new Random(1), other.learningRate);
        copyFrom(other);
    }

    public void copyFrom(Mlp o) {
        if (!Arrays.equals(this.sizes, o.sizes)) {
            throw new IllegalArgumentException("Cannot copy MLP with different topology");
        }

        this.learningRate = o.learningRate;
        this.beta1Pow = o.beta1Pow;
        this.beta2Pow = o.beta2Pow;

        for (int l = 0; l < weights.length; l++) {
            System.arraycopy(o.biases[l], 0, this.biases[l], 0, this.biases[l].length);
            System.arraycopy(o.mB[l], 0, this.mB[l], 0, this.mB[l].length);
            System.arraycopy(o.vB[l], 0, this.vB[l], 0, this.vB[l].length);

            for (int out = 0; out < weights[l].length; out++) {
                System.arraycopy(o.weights[l][out], 0, this.weights[l][out], 0, weights[l][out].length);
                System.arraycopy(o.mW[l][out], 0, this.mW[l][out], 0, mW[l][out].length);
                System.arraycopy(o.vW[l][out], 0, this.vW[l][out], 0, vW[l][out].length);
            }
        }
    }

    public float[] forward(float[] input) {
        return forward(input, 0);
    }

    public float[] forward(float[] input, int offset) {
        System.arraycopy(input, offset, acts[0], 0, sizes[0]);

        for (int l = 0; l < layers - 1; l++) {
            float[] in = acts[l];
            float[] out = acts[l + 1];
            float[] z = pre[l + 1];

            float[][] w = weights[l];
            float[] b = biases[l];

            for (int o = 0; o < out.length; o++) {
                float sum = b[o];
                float[] wo = w[o];
                for (int i = 0; i < in.length; i++) {
                    sum += wo[i] * in[i];
                }
                z[o] = sum;

                if (l == layers - 2) {
                    out[o] = sum; // linear output layer
                } else {
                    out[o] = Math.max(0.0f, sum); // ReLU hidden layers
                }
            }
        }

        return acts[layers - 1];
    }

    public void zeroGrad() {
        for (int l = 0; l < weights.length; l++) {
            Arrays.fill(gradB[l], 0.0f);
            for (int o = 0; o < gradW[l].length; o++) {
                Arrays.fill(gradW[l][o], 0.0f);
            }
        }
    }

    public void scaleGrad(float scale) {
        for (int l = 0; l < weights.length; l++) {
            for (int o = 0; o < gradB[l].length; o++) {
                gradB[l][o] *= scale;
                for (int i = 0; i < gradW[l][o].length; i++) {
                    gradW[l][o][i] *= scale;
                }
            }
        }
    }

    public void backward(float[] gradOutput) {
        System.arraycopy(gradOutput, 0, deltas[layers - 1], 0, gradOutput.length);

        for (int l = layers - 2; l >= 0; l--) {
            float[] d = deltas[l + 1];
            float[] in = acts[l];
            float[][] w = weights[l];

            for (int o = 0; o < d.length; o++) {
                float doVal = d[o];
                gradB[l][o] += doVal;

                float[] gw = gradW[l][o];
                for (int i = 0; i < in.length; i++) {
                    gw[i] += doVal * in[i];
                }
            }

            if (l > 0) {
                float[] dp = deltas[l];
                for (int i = 0; i < dp.length; i++) {
                    float sum = 0.0f;
                    for (int o = 0; o < d.length; o++) {
                        sum += w[o][i] * d[o];
                    }
                    dp[i] = pre[l][i] > 0.0f ? sum : 0.0f;
                }
            }
        }
    }

    public void step(float gradClip) {
        beta1Pow *= BETA1;
        beta2Pow *= BETA2;

        float corr1 = 1.0f - beta1Pow;
        float corr2 = 1.0f - beta2Pow;

        for (int l = 0; l < weights.length; l++) {
            for (int o = 0; o < weights[l].length; o++) {
                for (int i = 0; i < weights[l][o].length; i++) {
                    float g = gradW[l][o][i];
                    if (!Float.isFinite(g)) g = 0.0f;
                    if (g > gradClip) g = gradClip;
                    if (g < -gradClip) g = -gradClip;

                    float m = BETA1 * mW[l][o][i] + (1.0f - BETA1) * g;
                    float v = BETA2 * vW[l][o][i] + (1.0f - BETA2) * g * g;

                    mW[l][o][i] = m;
                    vW[l][o][i] = v;

                    float mHat = m / corr1;
                    float vHat = v / corr2;

                    weights[l][o][i] -= learningRate * mHat / ((float) Math.sqrt(vHat) + EPS);
                }

                float gb = gradB[l][o];
                if (!Float.isFinite(gb)) gb = 0.0f;
                if (gb > gradClip) gb = gradClip;
                if (gb < -gradClip) gb = -gradClip;

                float mb = BETA1 * mB[l][o] + (1.0f - BETA1) * gb;
                float vb = BETA2 * vB[l][o] + (1.0f - BETA2) * gb * gb;

                mB[l][o] = mb;
                vB[l][o] = vb;

                float mbHat = mb / corr1;
                float vbHat = vb / corr2;

                biases[l][o] -= learningRate * mbHat / ((float) Math.sqrt(vbHat) + EPS);
            }
        }
    }

    public boolean isFinite() {
        for (int l = 0; l < weights.length; l++) {
            for (int o = 0; o < weights[l].length; o++) {
                for (int i = 0; i < weights[l][o].length; i++) {
                    if (!Float.isFinite(weights[l][o][i])) return false;
                }
            }
            for (int o = 0; o < biases[l].length; o++) {
                if (!Float.isFinite(biases[l][o])) return false;
            }
        }
        return true;
    }

    public void write(DataOutputStream out) throws IOException {
        out.writeInt(sizes.length);
        for (int size : sizes) {
            out.writeInt(size);
        }

        out.writeFloat(learningRate);
        out.writeFloat(beta1Pow);
        out.writeFloat(beta2Pow);

        for (int l = 0; l < weights.length; l++) {
            for (int o = 0; o < weights[l].length; o++) {
                for (int i = 0; i < weights[l][o].length; i++) {
                    out.writeFloat(weights[l][o][i]);
                    out.writeFloat(mW[l][o][i]);
                    out.writeFloat(vW[l][o][i]);
                }
            }

            for (int o = 0; o < biases[l].length; o++) {
                out.writeFloat(biases[l][o]);
                out.writeFloat(mB[l][o]);
                out.writeFloat(vB[l][o]);
            }
        }
    }

    public static Mlp read(DataInputStream in) throws IOException {
        int layerCount = in.readInt();
        if (layerCount < 2 || layerCount > 8) {
            throw new IOException("Invalid MLP layer count: " + layerCount);
        }

        int[] sizes = new int[layerCount];
        for (int i = 0; i < layerCount; i++) {
            sizes[i] = in.readInt();
            if (sizes[i] <= 0 || sizes[i] > 4096) {
                throw new IOException("Invalid MLP layer size: " + sizes[i]);
            }
        }

        float lr = in.readFloat();
        Mlp mlp = new Mlp(sizes, new Random(1), lr);

        mlp.beta1Pow = in.readFloat();
        mlp.beta2Pow = in.readFloat();

        for (int l = 0; l < mlp.weights.length; l++) {
            for (int o = 0; o < mlp.weights[l].length; o++) {
                for (int i = 0; i < mlp.weights[l][o].length; i++) {
                    mlp.weights[l][o][i] = in.readFloat();
                    mlp.mW[l][o][i] = in.readFloat();
                    mlp.vW[l][o][i] = in.readFloat();
                }
            }

            for (int o = 0; o < mlp.biases[l].length; o++) {
                mlp.biases[l][o] = in.readFloat();
                mlp.mB[l][o] = in.readFloat();
                mlp.vB[l][o] = in.readFloat();
            }
        }

        return mlp;
    }
}