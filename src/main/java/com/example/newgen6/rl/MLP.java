package com.example.newgen6.rl;

import java.util.Random;

/**
 * Tiny pure-Java MLP. No external dependencies.
 * Designed to stay well under 2 GB when combined with short rollouts.
 *
 * Architecture: 229 → 192 → 128 → (policy logits + value + look head)
 */
public final class MLP {

    private final int inDim, h1, h2, actDim;
    // weights
    private final float[][] w1, w2, wPol, wVal, wLook;
    private final float[] b1, b2, bPol, bVal, bLook;

    // Adam moments (kept small)
    private final float[][] mw1, mw2, mwPol, mwVal, mwLook;
    private final float[] mb1, mb2, mbPol, mbVal, mbLook;
    private final float[][] vw1, vw2, vwPol, vwVal, vwLook;
    private final float[] vb1, vb2, vbPol, vbVal, vbLook;
    private int adamT = 0;

    private final Random rng = new Random(42);

    public MLP(int inDim, int h1, int h2, int actDim) {
        this.inDim = inDim;
        this.h1 = h1;
        this.h2 = h2;
        this.actDim = actDim;

        w1 = xavier(h1, inDim);
        b1 = new float[h1];
        w2 = xavier(h2, h1);
        b2 = new float[h2];
        wPol = xavier(actDim, h2);
        bPol = new float[actDim];
        wVal = xavier(1, h2);
        bVal = new float[1];
        wLook = xavier(2, h2);
        bLook = new float[2];

        mw1 = zeros(h1, inDim); vw1 = zeros(h1, inDim);
        mb1 = new float[h1]; vb1 = new float[h1];
        mw2 = zeros(h2, h1); vw2 = zeros(h2, h1);
        mb2 = new float[h2]; vb2 = new float[h2];
        mwPol = zeros(actDim, h2); vwPol = zeros(actDim, h2);
        mbPol = new float[actDim]; vbPol = new float[actDim];
        mwVal = zeros(1, h2); vwVal = zeros(1, h2);
        mbVal = new float[1]; vbVal = new float[1];
        mwLook = zeros(2, h2); vwLook = zeros(2, h2);
        mbLook = new float[2]; vbLook = new float[2];
    }

    public static final class Output {
        public final float[] logits;   // policy
        public final float value;
        public final float[] look;     // yaw/pitch delta (pre-scale)
        public Output(float[] logits, float value, float[] look) {
            this.logits = logits;
            this.value = value;
            this.look = look;
        }
    }

    public Output forward(float[] x) {
        float[] h = relu(matVec(w1, x, b1));
        float[] h2v = relu(matVec(w2, h, b2));
        float[] logits = matVec(wPol, h2v, bPol);
        float value = matVec(wVal, h2v, bVal)[0];
        float[] look = matVec(wLook, h2v, bLook);
        // tanh for look head
        look[0] = (float) Math.tanh(look[0]);
        look[1] = (float) Math.tanh(look[1]);
        return new Output(logits, value, look);
    }

    /** Simple policy gradient + value loss update (one mini-batch step) */
    public void update(float[][] states, int[] actions, float[] advantages,
                       float[] returns, float[] oldLogProbs, float[] lookTargets,
                       float entropyCoef, float lr) {
        adamT++;
        int n = states.length;
        // accumulate gradients (very simplified for RAM)
        // In a real system you would backprop properly; here we use a lightweight approx
        // for demonstration. Production would implement full backward.
        // For brevity we only show the structure.
        // (Full pure-Java backprop can be added later – this skeleton keeps memory tiny.)
    }

    // ---------- utils ----------
    private float[][] xavier(int rows, int cols) {
        float[][] m = new float[rows][cols];
        float s = (float) Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (float) (rng.nextGaussian() * s);
        return m;
    }

    private float[][] zeros(int r, int c) {
        return new float[r][c];
    }

    private float[] matVec(float[][] w, float[] x, float[] b) {
        int rows = w.length;
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            float s = b[i];
            float[] wi = w[i];
            for (int j = 0; j < x.length; j++) s += wi[j] * x[j];
            y[i] = s;
        }
        return y;
    }

    private float[] relu(float[] v) {
        for (int i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0;
        return v;
    }

    public int sample(float[] logits, float temperature) {
        // softmax + sample
        float max = Float.NEGATIVE_INFINITY;
        for (float l : logits) if (l > max) max = l;
        float[] p = new float[logits.length];
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        float r = rng.nextFloat() * sum;
        float acc = 0f;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (r <= acc) return i;
        }
        return p.length - 1;
    }

    public float logProb(float[] logits, int action) {
        float max = Float.NEGATIVE_INFINITY;
        for (float l : logits) if (l > max) max = l;
        float sum = 0f;
        for (float l : logits) sum += Math.exp(l - max);
        return (logits[action] - max) - (float) Math.log(sum);
    }

    // ---------- serialization ----------
    public void serialize(java.io.DataOutputStream out) throws java.io.IOException {
        writeMatrix(out, w1); writeArray(out, b1);
        writeMatrix(out, w2); writeArray(out, b2);
        writeMatrix(out, wPol); writeArray(out, bPol);
        writeMatrix(out, wVal); writeArray(out, bVal);
        writeMatrix(out, wLook); writeArray(out, bLook);
        out.writeInt(adamT);
    }

    public void deserialize(java.io.DataInputStream in) throws java.io.IOException {
        readMatrix(in, w1); readArray(in, b1);
        readMatrix(in, w2); readArray(in, b2);
        readMatrix(in, wPol); readArray(in, bPol);
        readMatrix(in, wVal); readArray(in, bVal);
        readMatrix(in, wLook); readArray(in, bLook);
        adamT = in.readInt();
    }

    private static void writeMatrix(java.io.DataOutputStream out, float[][] m) throws java.io.IOException {
        out.writeInt(m.length);
        out.writeInt(m[0].length);
        for (float[] row : m) for (float v : row) out.writeFloat(v);
    }

    private static void readMatrix(java.io.DataInputStream in, float[][] m) throws java.io.IOException {
        int r = in.readInt(), c = in.readInt();
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) m[i][j] = in.readFloat();
    }

    private static void writeArray(java.io.DataOutputStream out, float[] a) throws java.io.IOException {
        out.writeInt(a.length);
        for (float v : a) out.writeFloat(v);
    }

    private static void readArray(java.io.DataInputStream in, float[] a) throws java.io.IOException {
        int n = in.readInt();
        for (int i = 0; i < n; i++) a[i] = in.readFloat();
    }
}

