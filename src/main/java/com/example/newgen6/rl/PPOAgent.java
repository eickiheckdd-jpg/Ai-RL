package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

public class PPOAgent {
    // 16 Actor Outputs: 2 continuous (Pitch/Yaw) + 7 binary discrete heads (14 logits)
    private final int stateDim, hiddenDim, actionDim = 16;
    private final float lr = 0.00025f, gamma = 0.99f, gaeLambda = 0.95f, clipEps = 0.2f;
    private final float entropyCoeff = 0.01f; 

    private float actionStd = 0.6f; 
    private final float minActionStd = 0.05f;
    private final float decayRate = 0.999f;

    public Matrix w1, b1, wActor, bActor, wCritic, bCritic;
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
    }

    public static class StepData {
        public float[] state; 
        public float pitchDelta, yawDelta; 
        public int[] discreteActions; // 7 binary choices: [W, S, A, D, Jump, Sprint, Attack]
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

        // Tanh bounded continuous pitch/yaw deltas
        float muPitch = (float) Math.tanh(logits[0]) * 25f;
        float muYaw = (float) Math.tanh(logits[1]) * 35f;

        continuousOut[0] = (float) (muPitch + rng.nextGaussian() * actionStd);
        continuousOut[1] = (float) (muYaw + rng.nextGaussian() * actionStd);

        float logProbDiscreteAcc = 0.0f;
        
        // 7 Binary discrete heads: logits [2..15]
        for (int k = 0; k < 7; k++) {
            float[] probs = softmax(slice(logits, 2 + k * 2, 4 + k * 2));
            discreteOut[k] = sample(probs);
            logProbDiscreteAcc += (float) Math.log(probs[discreteOut[k]] + 1e-8f);
        }

        float logProbContinuous = -0.5f * (float) (
            Math.pow((continuousOut[0] - muPitch) / actionStd, 2) + 
            Math.pow((continuousOut[1] - muYaw) / actionStd, 2)
        ) - (float) Math.log(actionStd * Math.sqrt(2 * Math.PI));

        logProbOut[0] = logProbContinuous + logProbDiscreteAcc;
    }

    public void train(List<StepData> memory) {
        int N = memory.size();
        if (N == 0) return;

        float[] advantages = new float[N];
        float[] returns = new float[N];
        float gae = 0.0f;

        // Generalized Advantage Estimation (GAE)
        for (int i = N - 1; i >= 0; i--) {
            StepData cur = memory.get(i);
            float nextVal = (i == N - 1 || cur.done) ? 0.0f : memory.get(i + 1).value;
            float delta = cur.reward + gamma * nextVal - cur.value;
            gae = delta + gamma * gaeLambda * gae;
            advantages[i] = gae;
            returns[i] = gae + cur.value;
        }

        // PPO Optimization Epochs
        for (int epoch = 0; epoch < 4; epoch++) {
            for (int i = 0; i < N; i++) {
                StepData data = memory.get(i);

                float[] hiddenRaw = matmulAdd(w1, data.state, b1);
                float[] hidden = relu(hiddenRaw);
                float[] logits = matmulAdd(wActor, hidden, bActor);
                float valuePred = matmulAdd(wCritic, hidden, bCritic)[0];

                float muPitch = (float) Math.tanh(logits[0]) * 25f;
                float muYaw = (float) Math.tanh(logits[1]) * 35f;

                float curLogProbCont = -0.5f * (float) (
                    Math.pow((data.pitchDelta - muPitch) / actionStd, 2) + 
                    Math.pow((data.yawDelta - muYaw) / actionStd, 2)
                ) - (float) Math.log(actionStd * Math.sqrt(2 * Math.PI));

                float curLogProbDisc = 0.0f;
                float[][] probsDisc = new float[7][2];
                for (int k = 0; k < 7; k++) {
                    probsDisc[k] = softmax(slice(logits, 2 + k * 2, 4 + k * 2));
                    curLogProbDisc += (float) Math.log(probsDisc[k][data.discreteActions[k]] + 1e-8f);
                }

                float curLogProb = curLogProbCont + curLogProbDisc;
                float ratio = (float) Math.exp(curLogProb - data.logProb);
                float adv = advantages[i];

                // PPO Clipped Objective Gradient
                float dL_dLogProb;
                if ((ratio > 1f + clipEps && adv > 0) || (ratio < 1f - clipEps && adv < 0)) {
                    dL_dLogProb = 0.0f;
                } else {
                    dL_dLogProb = -adv * ratio;
                }

                // Logit Gradients (Actor)
                float[] dL_dLogits = new float[actionDim];

                // Continuous Pitch/Yaw Gradients through tanh
                float dLogProb_dMuP = (data.pitchDelta - muPitch) / (actionStd * actionStd);
                float dMuP_dZ0 = 25f * (1.0f - (float) Math.pow(Math.tanh(logits[0]), 2));
                dL_dLogits[0] = dL_dLogProb * dLogProb_dMuP * dMuP_dZ0;

                float dLogProb_dMuY = (data.yawDelta - muYaw) / (actionStd * actionStd);
                float dMuY_dZ1 = 35f * (1.0f - (float) Math.pow(Math.tanh(logits[1]), 2));
                dL_dLogits[1] = dL_dLogProb * dLogProb_dMuY * dMuY_dZ1;

                // Discrete Logit Gradients (Categorical Softmax + Entropy)
                for (int k = 0; k < 7; k++) {
                    int startIdx = 2 + k * 2;
                    int chosen = data.discreteActions[k];

                    for (int act = 0; act < 2; act++) {
                        float dLogProb_dLogit = (act == chosen ? 1.0f : 0.0f) - probsDisc[k][act];
                        float dEnt_dLogit = -probsDisc[k][act] * ((float) Math.log(probsDisc[k][act] + 1e-8f) + 1.0f);
                        dL_dLogits[startIdx + act] = dL_dLogProb * dLogProb_dLogit - entropyCoeff * dEnt_dLogit;
                    }
                }

                // Value Loss Gradient (Critic)
                float dL_dValue = valuePred - returns[i];

                // Backpropagate to Hidden Layer
                float[] dL_dHidden = new float[hiddenDim];
                for (int h = 0; h < hiddenDim; h++) {
                    float sum = 0.0f;
                    for (int a = 0; a < actionDim; a++) {
                        sum += wActor.data[a][h] * dL_dLogits[a];
                    }
                    sum += wCritic.data[0][h] * dL_dValue;
                    
                    // ReLU Derivative
                    dL_dHidden[h] = (hiddenRaw[h] > 0) ? sum : 0.0f;
                }

                // Update Actor Weights & Biases
                for (int a = 0; a < actionDim; a++) {
                    bActor.data[a][0] -= lr * dL_dLogits[a];
                    for (int h = 0; h < hiddenDim; h++) {
                        wActor.data[a][h] -= lr * dL_dLogits[a] * hidden[h];
                    }
                }

                // Update Critic Weights & Biases
                bCritic.data[0][0] -= lr * dL_dValue;
                for (int h = 0; h < hiddenDim; h++) {
                    wCritic.data[0][h] -= lr * dL_dValue * hidden[h];
                }

                // Update Input Layer Weights (w1) & Biases (b1)
                for (int h = 0; h < hiddenDim; h++) {
                    b1.data[h][0] -= lr * dL_dHidden[h];
                    for (int s = 0; s < stateDim; s++) {
                        w1.data[h][s] -= lr * dL_dHidden[h] * data.state[s];
                    }
                }
            }
        }

        // Decay exploration noise
        if (actionStd > minActionStd) {
            actionStd = Math.max(minActionStd, actionStd * decayRate);
        }
    }

    public void saveBrain(Path path) {
        try {
            Files.createDirectories(path.getParent());
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path.toFile()))) {
                dos.writeFloat(actionStd);
                writeMatrix(dos, w1); writeMatrix(dos, b1);
                writeMatrix(dos, wActor); writeMatrix(dos, bActor);
                writeMatrix(dos, wCritic); writeMatrix(dos, bCritic);
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    public void loadBrain(Path path) {
        if (!Files.exists(path)) return;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path.toFile()))) {
            actionStd = dis.readFloat();
            readMatrix(dis, w1); readMatrix(dis, b1);
            readMatrix(dis, wActor); readMatrix(dis, bActor);
            readMatrix(dis, wCritic); readMatrix(dis, bCritic);
            System.out.println("Loaded AI Brain! Exploration Noise (STD): " + actionStd);
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

    private int sample(float[] probs) {
        float r = rng.nextFloat(), cumulative = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1;
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

    private float[] softmax(float[] in) {
        float[] out = new float[in.length];
        float max = Float.NEGATIVE_INFINITY, sum = 0.0f;
        for (float v : in) if (v > max) max = v;
        for (int i = 0; i < in.length; i++) { out[i] = (float) Math.exp(in[i] - max); sum += out[i]; }
        for (int i = 0; i < in.length; i++) out[i] /= sum;
        return out;
    }
}
