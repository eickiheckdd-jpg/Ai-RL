package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

public class PPOAgent {
    private final int stateDim, hiddenDim = 128, actionDim = 16;
    private final float lr = 0.0003f, gamma = 0.99f, gaeLambda = 0.95f, clipEps = 0.2f;
    private final float entropyCoeff = 0.01f; 
    private final float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-5f;
    
    private int timeStep = 0;
    private int adamStep = 0;

    public Matrix wBase1, bBase1, wBase2, bBase2;
    public Matrix wActor, bActor;
    public Matrix wCritic1, bCritic1, wCritic2, bCritic2;

    private Matrix m_wB1, v_wB1, m_bB1, v_bB1, m_wB2, v_wB2, m_bB2, v_bB2;
    private Matrix m_wA, v_wA, m_bA, v_bA;
    private Matrix m_wC1, v_wC1, m_bC1, v_bC1, m_wC2, v_wC2, m_bC2, v_bC2;

    private final float[] logStd = new float[] { -0.5f, -0.5f };
    private final float[] m_logStd = new float[2];
    private final float[] v_logStd = new float[2];

    private final Random rng = new Random();

    public PPOAgent(int stateDim) {
        this.stateDim = stateDim;

        this.wBase1 = Matrix.randomOrthogonal(hiddenDim, stateDim, (float) Math.sqrt(2.0), rng);
        this.bBase1 = new Matrix(hiddenDim, 1);
        this.wBase2 = Matrix.randomOrthogonal(hiddenDim, hiddenDim, (float) Math.sqrt(2.0), rng);
        this.bBase2 = new Matrix(hiddenDim, 1);

        this.wActor = Matrix.randomOrthogonal(actionDim, hiddenDim, 0.01f, rng);
        this.bActor = new Matrix(actionDim, 1);

        this.wCritic1 = Matrix.randomOrthogonal(hiddenDim, stateDim, (float) Math.sqrt(2.0), rng);
        this.bCritic1 = new Matrix(hiddenDim, 1);
        this.wCritic2 = Matrix.randomOrthogonal(1, hiddenDim, 1.0f, rng);
        this.bCritic2 = new Matrix(1, 1);

        resetAdamBuffers();
    }

    private void resetAdamBuffers() {
        this.m_wB1 = new Matrix(hiddenDim, stateDim); this.v_wB1 = new Matrix(hiddenDim, stateDim);
        this.m_bB1 = new Matrix(hiddenDim, 1);        this.v_bB1 = new Matrix(hiddenDim, 1);
        this.m_wB2 = new Matrix(hiddenDim, hiddenDim); this.v_wB2 = new Matrix(hiddenDim, hiddenDim);
        this.m_bB2 = new Matrix(hiddenDim, 1);        this.v_bB2 = new Matrix(hiddenDim, 1);

        this.m_wA = new Matrix(actionDim, hiddenDim); this.v_wA = new Matrix(actionDim, hiddenDim);
        this.m_bA = new Matrix(actionDim, 1);        this.v_bA = new Matrix(actionDim, 1);

        this.m_wC1 = new Matrix(hiddenDim, stateDim); this.v_wC1 = new Matrix(hiddenDim, stateDim);
        this.m_bC1 = new Matrix(hiddenDim, 1);        this.v_bC1 = new Matrix(hiddenDim, 1);
        this.m_wC2 = new Matrix(1, hiddenDim);        this.v_wC2 = new Matrix(1, hiddenDim);
        this.m_bC2 = new Matrix(1, 1);                this.v_bC2 = new Matrix(1, 1);
    }

    public float[] getLogStd() { return logStd; }
    public int getTimeStep() { return timeStep; }

    public static class StepData {
        public float[] state; 
        public float pitchDelta, yawDelta; 
        public int[] discreteActions;
        public float logProb, reward, value; 
        public boolean done;

        public StepData(float[] s, float pd, float yd, int[] da, float lp, float r, float v, boolean d) {
            this.state = s.clone();
            this.pitchDelta = pd; 
            this.yawDelta = yd; 
            this.discreteActions = da.clone();
            this.logProb = lp; 
            this.reward = r; 
            this.value = v; 
            this.done = d;
        }
    }

    public synchronized void selectAction(float[] state, float[] continuousOut, int[] discreteOut, float[] logProbOut, float[] valueOut) {
        float[] h1 = leakyRelu(matmulAdd(wBase1, state, bBase1));
        float[] h2 = leakyRelu(matmulAdd(wBase2, h1, bBase2));
        float[] logits = matmulAdd(wActor, h2, bActor);

        float[] c1 = leakyRelu(matmulAdd(wCritic1, state, bCritic1));
        valueOut[0] = matmulAdd(wCritic2, c1, bCritic2)[0];

        float muPitch = logits[0];
        float muYaw = logits[1];

        float stdPitch = (float) Math.exp(logStd[0]);
        float stdYaw = (float) Math.exp(logStd[1]);

        float rawPitch = (float) (muPitch + rng.nextGaussian() * stdPitch);
        float rawYaw = (float) (muYaw + rng.nextGaussian() * stdYaw);

        continuousOut[0] = rawPitch;
        continuousOut[1] = rawYaw;

        float squashedPitch = (float) Math.tanh(rawPitch);
        float squashedYaw = (float) Math.tanh(rawYaw);

        float logProbDiscreteAcc = 0.0f;
        for (int k = 0; k < 7; k++) {
            float[] logProbsDisc = logSoftmax(slice(logits, 2 + k * 2, 4 + k * 2));
            discreteOut[k] = sampleFromLogProbs(logProbsDisc);
            logProbDiscreteAcc += logProbsDisc[discreteOut[k]];
        }

        float logProbPitch = -0.5f * (float) Math.pow((rawPitch - muPitch) / stdPitch, 2) 
                             - logStd[0] - 0.5f * (float) Math.log(2 * Math.PI)
                             - (float) Math.log(1.0f - squashedPitch * squashedPitch + 1e-6f);

        float logProbYaw   = -0.5f * (float) Math.pow((rawYaw - muYaw) / stdYaw, 2) 
                             - logStd[1] - 0.5f * (float) Math.log(2 * Math.PI)
                             - (float) Math.log(1.0f - squashedYaw * squashedYaw + 1e-6f);

        logProbOut[0] = logProbPitch + logProbYaw + logProbDiscreteAcc;
    }

    public synchronized void train(List<StepData> memory) {
        int N = memory.size();
        if (N == 0) return;

        timeStep++;
        float[] advantages = new float[N];
        float[] returns = new float[N];
        float gae = 0.0f;

        for (int i = N - 1; i >= 0; i--) {
            StepData cur = memory.get(i);
            if (cur.done) gae = 0.0f;
            float nextVal = (i == N - 1 || cur.done) ? 0.0f : memory.get(i + 1).value;
            float delta = cur.reward + gamma * nextVal - cur.value;
            gae = delta + gamma * gaeLambda * gae;
            advantages[i] = gae;
            returns[i] = gae + cur.value;
        }

        float advMean = 0.0f, advStd = 0.0f;
        for (float a : advantages) advMean += a;
        advMean /= N;
        for (float a : advantages) advStd += (a - advMean) * (a - advMean);
        advStd = (float) Math.sqrt((advStd / N) + 1e-5f);

        for (int i = 0; i < N; i++) advantages[i] = (advantages[i] - advMean) / advStd;

        for (int epoch = 0; epoch < 4; epoch++) {
            adamStep++;

            Matrix g_wB1 = new Matrix(hiddenDim, stateDim);    Matrix g_bB1 = new Matrix(hiddenDim, 1);
            Matrix g_wB2 = new Matrix(hiddenDim, hiddenDim);  Matrix g_bB2 = new Matrix(hiddenDim, 1);
            Matrix g_wA  = new Matrix(actionDim, hiddenDim);  Matrix g_bA  = new Matrix(actionDim, 1);

            Matrix g_wC1 = new Matrix(hiddenDim, stateDim);   Matrix g_bC1 = new Matrix(hiddenDim, 1);
            Matrix g_wC2 = new Matrix(1, hiddenDim);         Matrix g_bC2 = new Matrix(1, 1);

            float[] g_logStd = new float[2];

            float stdPitch = (float) Math.exp(logStd[0]);
            float stdYaw = (float) Math.exp(logStd[1]);

            for (int i = 0; i < N; i++) {
                StepData data = memory.get(i);

                float[] h1Raw = matmulAdd(wBase1, data.state, bBase1);
                float[] h1 = leakyRelu(h1Raw);
                float[] h2Raw = matmulAdd(wBase2, h1, bBase2);
                float[] h2 = leakyRelu(h2Raw);
                float[] logits = matmulAdd(wActor, h2, bActor);

                float[] c1Raw = matmulAdd(wCritic1, data.state, bCritic1);
                float[] c1 = leakyRelu(c1Raw);
                float valuePred = matmulAdd(wCritic2, c1, bCritic2)[0];

                float muPitch = logits[0];
                float muYaw = logits[1];

                float rawPitch = data.pitchDelta;
                float rawYaw = data.yawDelta;

                float squashedPitch = (float) Math.tanh(rawPitch);
                float squashedYaw = (float) Math.tanh(rawYaw);

                float curLogProbPitch = -0.5f * (float) Math.pow((rawPitch - muPitch) / stdPitch, 2) 
                                     - logStd[0] - 0.5f * (float) Math.log(2 * Math.PI)
                                     - (float) Math.log(1.0f - squashedPitch * squashedPitch + 1e-6f);

                float curLogProbYaw   = -0.5f * (float) Math.pow((rawYaw - muYaw) / stdYaw, 2) 
                                     - logStd[1] - 0.5f * (float) Math.log(2 * Math.PI)
                                     - (float) Math.log(1.0f - squashedYaw * squashedYaw + 1e-6f);

                float curLogProbDisc = 0.0f;
                float[][] logProbsDisc = new float[7][2];
                for (int k = 0; k < 7; k++) {
                    logProbsDisc[k] = logSoftmax(slice(logits, 2 + k * 2, 4 + k * 2));
                    curLogProbDisc += logProbsDisc[k][data.discreteActions[k]];
                }

                float curLogProb = curLogProbPitch + curLogProbYaw + curLogProbDisc;
                float ratio = (float) Math.exp(curLogProb - data.logProb);
                float adv = advantages[i];

                float dL_dLogProb = ((ratio > 1f + clipEps && adv > 0) || (ratio < 1f - clipEps && adv < 0)) ? 0.0f : -adv * ratio;

                float[] dL_dLogits = new float[actionDim];

                float diffP = (rawPitch - muPitch) / stdPitch;
                float diffY = (rawYaw - muYaw) / stdYaw;

                dL_dLogits[0] = dL_dLogProb * (diffP / stdPitch);
                dL_dLogits[1] = dL_dLogProb * (diffY / stdYaw);

                g_logStd[0] += dL_dLogProb * (diffP * diffP - 1.0f) - entropyCoeff;
                g_logStd[1] += dL_dLogProb * (diffY * diffY - 1.0f) - entropyCoeff;

                for (int k = 0; k < 7; k++) {
                    int startIdx = 2 + k * 2;
                    int chosen = data.discreteActions[k];
                    for (int act = 0; act < 2; act++) {
                        float pAct = (float) Math.exp(logProbsDisc[k][act]);
                        float dLogProb_dLogit = (act == chosen ? 1.0f : 0.0f) - pAct;
                        dL_dLogits[startIdx + act] += dL_dLogProb * dLogProb_dLogit;
                    }
                }

                float dL_dValue = 0.5f * (valuePred - returns[i]);
                float[] dL_dC1 = new float[hiddenDim];
                for (int h = 0; h < hiddenDim; h++) {
                    dL_dC1[h] = (c1Raw[h] > 0) ? (wCritic2.data[0][h] * dL_dValue) : 0.01f * (wCritic2.data[0][h] * dL_dValue);
                }

                for (int h = 0; h < hiddenDim; h++) {
                    g_bC2.data[0][0] += dL_dValue / N;
                    g_wC2.data[0][h] += (dL_dValue * c1[h]) / N;
                    g_bC1.data[h][0] += dL_dC1[h] / N;
                    for (int s = 0; s < stateDim; s++) g_wC1.data[h][s] += (dL_dC1[h] * data.state[s]) / N;
                }

                float[] dL_dH2 = new float[hiddenDim];
                for (int h = 0; h < hiddenDim; h++) {
                    float sum = 0.0f;
                    for (int a = 0; a < actionDim; a++) sum += wActor.data[a][h] * dL_dLogits[a];
                    dL_dH2[h] = (h2Raw[h] > 0) ? sum : 0.01f * sum;
                }

                float[] dL_dH1 = new float[hiddenDim];
                for (int h = 0; h < hiddenDim; h++) {
                    float sum = 0.0f;
                    for (int h2Idx = 0; h2Idx < hiddenDim; h2Idx++) sum += wBase2.data[h2Idx][h] * dL_dH2[h2Idx];
                    dL_dH1[h] = (h1Raw[h] > 0) ? sum : 0.01f * sum;
                }

                for (int a = 0; a < actionDim; a++) {
                    g_bA.data[a][0] += dL_dLogits[a] / N;
                    for (int h = 0; h < hiddenDim; h++) g_wA.data[a][h] += (dL_dLogits[a] * h2[h]) / N;
                }

                for (int h = 0; h < hiddenDim; h++) {
                    g_bB2.data[h][0] += dL_dH2[h] / N;
                    for (int prev = 0; prev < hiddenDim; prev++) g_wB2.data[h][prev] += (dL_dH2[h] * h1[prev]) / N;

                    g_bB1.data[h][0] += dL_dH1[h] / N;
                    for (int s = 0; s < stateDim; s++) g_wB1.data[h][s] += (dL_dH1[h] * data.state[s]) / N;
                }
            }

            clipGlobalNorm(new Matrix[]{g_wB1, g_bB1, g_wB2, g_bB2, g_wA, g_bA, g_wC1, g_bC1, g_wC2, g_bC2}, 0.5f);

            wBase1.adamUpdate(g_wB1, m_wB1, v_wB1, adamStep, lr, beta1, beta2, eps);
            bBase1.adamUpdate(g_bB1, m_bB1, v_bB1, adamStep, lr, beta1, beta2, eps);
            wBase2.adamUpdate(g_wB2, m_wB2, v_wB2, adamStep, lr, beta1, beta2, eps);
            bBase2.adamUpdate(g_bB2, m_bB2, v_bB2, adamStep, lr, beta1, beta2, eps);

            wActor.adamUpdate(g_wA, m_wA, v_wA, adamStep, lr, beta1, beta2, eps);
            bActor.adamUpdate(g_bA, m_bA, v_bA, adamStep, lr, beta1, beta2, eps);

            wCritic1.adamUpdate(g_wC1, m_wC1, v_wC1, adamStep, lr, beta1, beta2, eps);
            bCritic1.adamUpdate(g_bC1, m_bC1, v_bC1, adamStep, lr, beta1, beta2, eps);
            wCritic2.adamUpdate(g_wC2, m_wC2, v_wC2, adamStep, lr, beta1, beta2, eps);
            bCritic2.adamUpdate(g_bC2, m_bC2, v_bC2, adamStep, lr, beta1, beta2, eps);

            for (int idx = 0; idx < 2; idx++) {
                float g = g_logStd[idx] / N;
                m_logStd[idx] = beta1 * m_logStd[idx] + (1.0f - beta1) * g;
                v_logStd[idx] = beta2 * v_logStd[idx] + (1.0f - beta2) * (g * g);
                float mHat = m_logStd[idx] / (1.0f - (float) Math.pow(beta1, adamStep));
                float vHat = v_logStd[idx] / (1.0f - (float) Math.pow(beta2, adamStep));
                logStd[idx] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
                logStd[idx] = Math.max(-2.0f, Math.min(0.5f, logStd[idx]));
            }
        }
    }

    private void clipGlobalNorm(Matrix[] matrices, float maxNorm) {
        float sumNormSq = 0.0f;
        for (Matrix m : matrices) {
            for (int i = 0; i < m.rows; i++) {
                for (int j = 0; j < m.cols; j++) sumNormSq += m.data[i][j] * m.data[i][j];
            }
        }
        float globalNorm = (float) Math.sqrt(sumNormSq);
        if (globalNorm > maxNorm) {
            float scale = maxNorm / globalNorm;
            for (Matrix m : matrices) {
                for (int i = 0; i < m.rows; i++) {
                    for (int j = 0; j < m.cols; j++) m.data[i][j] *= scale;
                }
            }
        }
    }

    public synchronized void saveBrain(Path path) {
        try {
            Files.createDirectories(path.getParent());
            try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path.toFile())))) {
                dos.writeInt(timeStep); dos.writeInt(adamStep);
                dos.writeFloat(logStd[0]); dos.writeFloat(logStd[1]);
                writeMatrix(dos, wBase1); writeMatrix(dos, bBase1);
                writeMatrix(dos, wBase2); writeMatrix(dos, bBase2);
                writeMatrix(dos, wActor); writeMatrix(dos, bActor);
                writeMatrix(dos, wCritic1); writeMatrix(dos, bCritic1);
                writeMatrix(dos, wCritic2); writeMatrix(dos, bCritic2);
                dos.flush();
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    public synchronized void loadBrain(Path path) {
        if (!Files.exists(path)) return;
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path.toFile())))) {
            timeStep = dis.readInt(); adamStep = dis.readInt();
            logStd[0] = dis.readFloat(); logStd[1] = dis.readFloat();
            readMatrix(dis, wBase1); readMatrix(dis, bBase1);
            readMatrix(dis, wBase2); readMatrix(dis, bBase2);
            readMatrix(dis, wActor); readMatrix(dis, bActor);
            readMatrix(dis, wCritic1); readMatrix(dis, bCritic1);
            readMatrix(dis, wCritic2); readMatrix(dis, bCritic2);
            resetAdamBuffers();
        } catch (IOException e) { e.printStackTrace(); }
    }

    private void writeMatrix(DataOutputStream dos, Matrix m) throws IOException {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) dos.writeFloat(m.data[i][j]);
        }
    }

    private void readMatrix(DataInputStream dis, Matrix m) throws IOException {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) m.data[i][j] = dis.readFloat();
        }
    }

    private int sampleFromLogProbs(float[] logProbs) {
        float r = rng.nextFloat();
        float cumulative = 0.0f;
        for (int i = 0; i < logProbs.length; i++) {
            cumulative += (float) Math.exp(logProbs[i]);
            if (r <= cumulative) return i;
        }
        return logProbs.length - 1;
    }

    private float[] logSoftmax(float[] in) {
        float[] out = new float[in.length];
        float max = Float.NEGATIVE_INFINITY;
        for (float v : in) if (v > max) max = v;

        float sumExp = 0.0f;
        for (float v : in) sumExp += (float) Math.exp(v - max);
        float logSumExp = max + (float) Math.log(sumExp);

        for (int i = 0; i < in.length; i++) out[i] = in[i] - logSumExp;
        return out;
    }

    private float[] slice(float[] arr, int start, int end) {
        float[] res = new float[end - start];
        System.arraycopy(arr, start, res, 0, end - start);
        return res;
    }

    private float[] matmulAdd(Matrix w, float[] x, Matrix b) {
        float[] out = new float[w.rows];
        for (int i = 0; i < w.rows; i++) {
            out[i] = b.data[i][0];
            for (int j = 0; j < w.cols; j++) out[i] += w.data[i][j] * x[j];
        }
        return out;
    }

    private float[] leakyRelu(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) out[i] = in[i] > 0 ? in[i] : 0.01f * in[i];
        return out;
    }
}
