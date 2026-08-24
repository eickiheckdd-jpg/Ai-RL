package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

public class PPOAgent {
    private final int stateDim, hiddenDim, actionDim = 11;
    private final float lr = 0.00025f, gamma = 0.99f, gaeLambda = 0.95f, clipEps = 0.2f;
    private final float entropyCoeff = 0.01f; // Prevents policy collapse
    
    // Exploration Decay
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

        // Tanh bounded continuous aim output
        float muPitch = (float) Math.tanh(logits[0]) * 25f;
        float muYaw = (float) Math.tanh(logits[1]) * 35f;
        
        continuousOut[0] = (float) (muPitch + rng.nextGaussian() * actionStd);
        continuousOut[1] = (float) (muYaw + rng.nextGaussian() * actionStd);

        float[] pMove = softmax(slice(logits, 2, 7));
        float[] pJump = softmax(slice(logits, 7, 9));
        float[] pAttack = softmax(slice(logits, 9, 11));

        discreteOut[0] = sample(pMove);
        discreteOut[1] = sample(pJump);
        discreteOut[2] = sample(pAttack);

        float logProbContinuous = -0.5f * (float)(
            Math.pow((continuousOut[0] - muPitch) / actionStd, 2) + 
            Math.pow((continuousOut[1] - muYaw) / actionStd, 2)
        );
        
        logProbOut[0] = logProbContinuous + (float) (
            Math.log(pMove[discreteOut[0]] + 1e-8) + 
            Math.log(pJump[discreteOut[1]] + 1e-8) + 
            Math.log(pAttack[discreteOut[2]] + 1e-8)
        );
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
                
                float[] hidden = relu(matmulAdd(w1, data.state, b1));
                float[] logits = matmulAdd(wActor, hidden, bActor);
                float valuePred = matmulAdd(wCritic, hidden, bCritic)[0];
                
                float muPitch = (float) Math.tanh(logits[0]) * 25f;
                float muYaw = (float) Math.tanh(logits[1]) * 35f;
                
                float[] pMove = softmax(slice(logits, 2, 7));
                float[] pJump = softmax(slice(logits, 7, 9));
                float[] pAttack = softmax(slice(logits, 9, 11));

                // Entropy Calculation (encourages exploration)
                float entropy = calculateEntropy(pMove) + calculateEntropy(pJump) + calculateEntropy(pAttack);

                float curLogProb = -0.5f * (float)(
                    Math.pow((data.pitchDelta - muPitch) / actionStd, 2) + 
                    Math.pow((data.yawDelta - muYaw) / actionStd, 2)
                ) + (float) (
                    Math.log(pMove[data.discreteActions[0]] + 1e-8) + 
                    Math.log(pJump[data.discreteActions[1]] + 1e-8) + 
                    Math.log(pAttack[data.discreteActions[2]] + 1e-8)
                );

                float ratio = (float) Math.exp(curLogProb - data.logProb);
                float adv = advantages[i];
                
                // Clipped Policy Objective Gradient
                float policyGradMul = (ratio > 1 + clipEps && adv > 0) || (ratio < 1 - clipEps && adv < 0) ? 0.0f : -adv * ratio;
                policyGradMul -= entropyCoeff * entropy; // Inject entropy gradient
                
                float valueGrad = (valuePred - returns[i]);

                // Update Actor Weights
                for (int a = 0; a < actionDim; a++) {
                    float gradA = policyGradMul * 0.01f;
                    bActor.data[a][0] -= lr * gradA;
                    for (int h = 0; h < hiddenDim; h++) {
                        wActor.data[a][h] -= lr * gradA * hidden[h];
                    }
                }

                // Update Critic Weights
                bCritic.data[0][0] -= lr * valueGrad * 0.01f;
                for (int h = 0; h < hiddenDim; h++) {
                    wCritic.data[0][h] -= lr * valueGrad * hidden[h];
                }

                // Update Hidden Layer (w1)
                for (int h = 0; h < hiddenDim; h++) {
                    if (hidden[h] > 0) { 
                        for (int s = 0; s < stateDim; s++) {
                            w1.data[h][s] -= lr * policyGradMul * data.state[s] * 0.001f;
                        }
                    }
                }
            }
        }

        // Decay Exploration Noise over time
        if (actionStd > minActionStd) {
            actionStd = Math.max(minActionStd, actionStd * decayRate);
        }
    }

    private float calculateEntropy(float[] probs) {
        float ent = 0.0f;
        for (float p : probs) {
            if (p > 1e-8f) ent -= p * (float) Math.log(p);
        }
        return ent;
    }

    public void saveBrain(Path path) {
        try {
            Files.createDirectories(path.getParent());
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path.toFile()))) {
                dos.writeFloat(actionStd); // Save current noise level
                writeMatrix(dos, w1); writeMatrix(dos, b1);
                writeMatrix(dos, wActor); writeMatrix(dos, bActor);
                writeMatrix(dos, wCritic); writeMatrix(dos, bCritic);
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    public void loadBrain(Path path) {
        if (!Files.exists(path)) return;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path.toFile()))) {
            actionStd = dis.readFloat(); // Load saved noise level
            readMatrix(dis, w1); readMatrix(dis, b1);
            readMatrix(dis, wActor); readMatrix(dis, bActor);
            readMatrix(dis, wCritic); readMatrix(dis, bCritic);
            System.out.println("Loaded AI Brain! Current Exploration Noise (STD): " + actionStd);
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
