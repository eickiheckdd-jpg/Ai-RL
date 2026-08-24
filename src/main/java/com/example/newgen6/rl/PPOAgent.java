package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

public class PPOAgent {
    private final int stateDim, hiddenDim, actionDim = 16;
    private final float lr = 0.0003f, gamma = 0.99f, gaeLambda = 0.95f, clipEps = 0.2f;
    private final float entropyCoeff = 0.01f; 

    private final float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    private int timeStep = 0;

    public Matrix w1, b1, wActor, bActor, wCritic, bCritic;
    
    private Matrix m_w1, v_w1, m_b1, v_b1;
    private Matrix m_wA, v_wA, m_bA, v_bA;
    private Matrix m_wC, v_wC, m_bC, v_bC;

    private final float[] logStd = new float[] { -0.5f, -0.5f };
    private final float[] m_logStd = new float[2];
    private final float[] v_logStd = new float[2];

    private final Random rng = new Random();

    public PPOAgent(int stateDim, int hiddenDim) {
        this.stateDim = stateDim;
        this.hiddenDim = hiddenDim;

        this.w1 = Matrix.randomXavier(hiddenDim, stateDim, rng);
        this.b1 = new Matrix(hiddenDim, 1);
        this.wActor = Matrix.randomXavier(actionDim, hiddenDim, rng);
        this.bActor = new Matrix(actionDim, 1);
        this.wCritic = Matrix.randomXavier(1, hiddenDim, rng);
        this.bCritic = new Matrix(1, 1);

        resetAdamBuffers();
    }

    private void resetAdamBuffers() {
        this.m_w1 = new Matrix(hiddenDim, stateDim); this.v_w1 = new Matrix(hiddenDim, stateDim);
        this.m_b1 = new Matrix(hiddenDim, 1);        this.v_b1 = new Matrix(hiddenDim, 1);
        this.m_wA = new Matrix(actionDim, hiddenDim); this.v_wA = new Matrix(actionDim, hiddenDim);
        this.m_bA = new Matrix(actionDim, 1);        this.v_bA = new Matrix(actionDim, 1);
        this.m_wC = new Matrix(1, hiddenDim);        this.v_wC = new Matrix(1, hiddenDim);
        this.m_bC = new Matrix(1, 1);                this.v_bC = new Matrix(1, 1);
    }

    // --- Added Getters for HUD ---
    public float[] getLogStd() {
        return logStd;
    }

    public int getTimeStep() {
        return timeStep;
    }
    // ----------------------------

    public static class StepData {
        public float[] state; 
        public float pitchDelta, yawDelta; 
        public int[] discreteActions;
        public float logProb, reward, value; 
        public boolean done;

        public StepData(float[] s, float pd, float yd, int[] da, float lp, float r, float v, boolean d) {
            this.state = s; this.pitchDelta = pd; this.yawDelta = yd; this.discreteActions = da;
            this.logProb = lp; this.reward = r; this.value = v; this.done = d;
        }
    }

    public void selectAction(float[] state, float[] continuousOut, int[] discreteOut, float[] logProbOut, float[] valueOut) {
        float[] hidden = relu(matmulAdd(w1, state, b1));
        float[] logits = matmulAdd(wActor, hidden, bActor);
        valueOut[0] = matmulAdd(wCritic, hidden, bCritic)[0];

        float muPitch = (float) Math.tanh(logits[0]);
        float muYaw = (float) Math.tanh(logits[1]);

        float stdPitch = (float) Math.exp(logStd[0]);
        float stdYaw = (float) Math.exp(logStd[1]);

        continuousOut[0] = (float) (muPitch + rng.nextGaussian() * stdPitch);
        continuousOut[1] = (float) (muYaw + rng.nextGaussian() * stdYaw);

        float logProbDiscreteAcc = 0.0f;
        for (int k = 0; k < 7; k++) {
            float[] logProbsDisc = logSoftmax(slice(logits, 2 + k * 2, 4 + k * 2));
            discreteOut[k] = sampleFromLogProbs(logProbsDisc);
            logProbDiscreteAcc += logProbsDisc[discreteOut[k]];
        }

        float logProbPitch = -0.5f * (float) Math.pow((continuousOut[0] - muPitch) / stdPitch, 2) - logStd[0] - 0.5f * (float) Math.log(2 * Math.PI);
        float logProbYaw   = -0.5f * (float) Math.pow((continuousOut[1] - muYaw) / stdYaw, 2) - logStd[1] - 0.5f * (float) Math.log(2 * Math.PI);

        logProbOut[0] = logProbPitch + logProbYaw + logProbDiscreteAcc;
    }

    public void train(List<StepData> memory) {
        int N = memory.size();
        if (N == 0) return;

        timeStep++;
        float[] advantages = new float[N];
        float[] returns = new float[N];
        float gae = 0.0f;

        for (int i = N - 1; i >= 0; i--) {
            StepData cur = memory.get(i);
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
        advStd = (float) Math.sqrt(advStd / N) + 1e-8f;

        for (int i = 0; i < N; i++) {
            advantages[i] = (advantages[i] - advMean) / advStd;
        }

        for (int epoch = 0; epoch < 4; epoch++) {
            Matrix g_w1 = new Matrix(hiddenDim, stateDim);
            Matrix g_b1 = new Matrix(hiddenDim, 1);
            Matrix g_wA = new Matrix(actionDim, hiddenDim);
            Matrix g_bA = new Matrix(actionDim, 1);
            Matrix g_wC = new Matrix(1, hiddenDim);
            Matrix g_bC = new Matrix(1, 1);
            float[] g_logStd = new float[2];

            float stdPitch = (float) Math.exp(logStd[0]);
            float stdYaw = (float) Math.exp(logStd[1]);

            for (int i = 0; i < N; i++) {
                StepData data = memory.get(i);

                float[] hiddenRaw = matmulAdd(w1, data.state, b1);
                float[] hidden = relu(hiddenRaw);
                float[] logits = matmulAdd(wActor, hidden, bActor);
                float valuePred = matmulAdd(wCritic, hidden, bCritic)[0];

                float muPitch = (float) Math.tanh(logits[0]);
                float muYaw = (float) Math.tanh(logits[1]);

                float curLogProbPitch = -0.5f * (float) Math.pow((data.pitchDelta - muPitch) / stdPitch, 2) - logStd[0] - 0.5f * (float) Math.log(2 * Math.PI);
                float curLogProbYaw   = -0.5f * (float) Math.pow((data.yawDelta - muYaw) / stdYaw, 2) - logStd[1] - 0.5f * (float) Math.log(2 * Math.PI);

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

                float dLogProb_dMuP = (data.pitchDelta - muPitch) / (stdPitch * stdPitch);
                float dMuP_dZ0 = 1.0f - (muPitch * muPitch);
                dL_dLogits[0] = dL_dLogProb * dLogProb_dMuP * dMuP_dZ0;

                float dLogProb_dMuY = (data.yawDelta - muYaw) / (stdYaw * stdYaw);
                float dMuY_dZ1 = 1.0f - (muYaw * muYaw);
                dL_dLogits[1] = dL_dLogProb * dLogProb_dMuY * dMuY_dZ1;

                g_logStd[0] += dL_dLogProb * ((float) Math.pow((data.pitchDelta - muPitch) / stdPitch, 2) - 1.0f);
                g_logStd[1] += dL_dLogProb * ((float) Math.pow((data.yawDelta - muYaw) / stdYaw, 2) - 1.0f);

                for (int k = 0; k < 7; k++) {
                    int startIdx = 2 + k * 2;
                    int chosen = data.discreteActions[k];
                    
                    float entropy = 0.0f;
                    for (int act = 0; act < 2; act++) {
                        float pAct = (float) Math.exp(logProbsDisc[k][act]);
                        entropy -= pAct * logProbsDisc[k][act];
                    }

                    for (int act = 0; act < 2; act++) {
                        float pAct = (float) Math.exp(logProbsDisc[k][act]);
                        float dLogProb_dLogit = (act == chosen ? 1.0f : 0.0f) - pAct;
                        float dEnt_dLogit = -pAct * (logProbsDisc[k][act] + entropy);
                        dL_dLogits[startIdx + act] = dL_dLogProb * dLogProb_dLogit - entropyCoeff * dEnt_dLogit;
                    }
                }

                float dL_dValue = 0.5f * (valuePred - returns[i]);

                float[] dL_dHidden = new float[hiddenDim];
                for (int h = 0; h < hiddenDim; h++) {
                    float sum = 0.0f;
                    for (int a = 0; a < actionDim; a++) sum += wActor.data[a][h] * dL_dLogits[a];
                    sum += wCritic.data[0][h] * dL_dValue;
                    dL_dHidden[h] = (hiddenRaw[h] > 0) ? sum : 0.0f;
                }

                for (int a = 0; a < actionDim; a++) {
                    g_bA.data[a][0] += dL_dLogits[a] / N;
                    for (int h = 0; h < hiddenDim; h++) g_wA.data[a][h] += (dL_dLogits[a] * hidden[h]) / N;
                }

                g_bC.data[0][0] += dL_dValue / N;
                for (int h = 0; h < hiddenDim; h++) g_wC.data[0][h] += (dL_dValue * hidden[h]) / N;

                for (int h = 0; h < hiddenDim; h++) {
                    g_b1.data[h][0] += dL_dHidden[h] / N;
                    for (int s = 0; s < stateDim; s++) g_w1.data[h][s] += (dL_dHidden[h] * data.state[s]) / N;
                }
            }

            clipMatrixGradients(g_w1, 1.0f);
            clipMatrixGradients(g_wA, 1.0f);
            clipMatrixGradients(g_wC, 1.0f);

            w1.adamUpdate(g_w1, m_w1, v_w1, timeStep, lr, beta1, beta2, eps);
            b1.adamUpdate(g_b1, m_b1, v_b1, timeStep, lr, beta1, beta2, eps);
            wActor.adamUpdate(g_wA, m_wA, v_wA, timeStep, lr, beta1, beta2, eps);
            bActor.adamUpdate(g_bA, m_bA, v_bA, timeStep, lr, beta1, beta2, eps);
            wCritic.adamUpdate(g_wC, m_wC, v_wC, timeStep, lr, beta1, beta2, eps);
            bCritic.adamUpdate(g_bC, m_bC, v_bC, timeStep, lr, beta1, beta2, eps);

            for (int idx = 0; idx < 2; idx++) {
                float g = g_logStd[idx] / N;
                m_logStd[idx] = beta1 * m_logStd[idx] + (1.0f - beta1) * g;
                v_logStd[idx] = beta2 * v_logStd[idx] + (1.0f - beta2) * (g * g);
                float mHat = m_logStd[idx] / (1.0f - (float) Math.pow(beta1, timeStep));
                float vHat = v_logStd[idx] / (1.0f - (float) Math.pow(beta2, timeStep));
                logStd[idx] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
                logStd[idx] = Math.max(-2.0f, Math.min(0.5f, logStd[idx]));
            }
        }
    }

    public void saveBrain(Path path) {
        try {
            Files.createDirectories(path.getParent());
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path.toFile()))) {
                dos.writeInt(timeStep);
                dos.writeFloat(logStd[0]); dos.writeFloat(logStd[1]);
                writeMatrix(dos, w1); writeMatrix(dos, b1);
                writeMatrix(dos, wActor); writeMatrix(dos, bActor);
                writeMatrix(dos, wCritic); writeMatrix(dos, bCritic);
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    public void loadBrain(Path path) {
        if (!Files.exists(path)) return;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path.toFile()))) {
            timeStep = dis.readInt();
            logStd[0] = dis.readFloat(); logStd[1] = dis.readFloat();
            readMatrix(dis, w1); readMatrix(dis, b1);
            readMatrix(dis, wActor); readMatrix(dis, bActor);
            readMatrix(dis, wCritic); readMatrix(dis, bCritic);
            
            resetAdamBuffers();
            System.out.println("Loaded AI Brain with resynced Adam states! Epoch: " + timeStep);
        } catch (IOException e) { e.printStackTrace(); }
    }

    private void clipMatrixGradients(Matrix g, float maxVal) {
        for (int i = 0; i < g.rows; i++) {
            for (int j = 0; j < g.cols; j++) {
                g.data[i][j] = Math.max(-maxVal, Math.min(maxVal, g.data[i][j]));
            }
        }
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

        for (int i = 0; i < in.length; i++) {
            out[i] = in[i] - logSumExp;
        }
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

    private float[] relu(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) out[i] = Math.max(0, in[i]);
        return out;
    }
}
