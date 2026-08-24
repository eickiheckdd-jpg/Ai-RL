package com.example.newgen6.rl;

import java.util.List;
import java.util.Random;

public class PPOAgent {
    private final int stateDim, hiddenDim;
    private final float lr = 0.0003f, gamma = 0.99f, clipEps = 0.2f;
    private final float ACTION_STD = 0.5f; // Fixed standard deviation for continuous exploration
    
    public Matrix w1, b1, wActor, bActor, wCritic, bCritic;
    private final Random rng = new Random();

    public PPOAgent(int stateDim, int hiddenDim) {
        this.stateDim = stateDim;
        this.hiddenDim = hiddenDim;
        
        // Output dim: 2 (Continuous Pitch/Yaw) + 5 (Move) + 2 (Jump) + 2 (Attack) = 11
        int outDim = 11; 

        this.w1 = Matrix.randomXavier(hiddenDim, stateDim, rng);
        this.b1 = new Matrix(hiddenDim, 1);
        this.wActor = Matrix.randomXavier(outDim, hiddenDim, rng);
        this.bActor = new Matrix(outDim, 1);
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

        // Continuous Heads (Index 0, 1 are Mu for Pitch and Yaw)
        float muPitch = (float) Math.tanh(logits[0]) * 30f; // Constrain delta to -30 to +30 degrees
        float muYaw = (float) Math.tanh(logits[1]) * 40f;
        
        continuousOut[0] = (float) (muPitch + rng.nextGaussian() * ACTION_STD);
        continuousOut[1] = (float) (muYaw + rng.nextGaussian() * ACTION_STD);

        // Discrete Heads
        float[] pMove = softmax(slice(logits, 2, 7));
        float[] pJump = softmax(slice(logits, 7, 9));
        float[] pAttack = softmax(slice(logits, 9, 11));

        discreteOut[0] = sample(pMove);
        discreteOut[1] = sample(pJump);
        discreteOut[2] = sample(pAttack);

        // Log Probability (Simplified combination of Gaussian PDF and Discrete Logits)
        float logProbContinuous = -0.5f * (float)(
            Math.pow((continuousOut[0] - muPitch) / ACTION_STD, 2) + 
            Math.pow((continuousOut[1] - muYaw) / ACTION_STD, 2)
        );
        
        logProbOut[0] = logProbContinuous + (float) (
            Math.log(pMove[discreteOut[0]] + 1e-8) + 
            Math.log(pJump[discreteOut[1]] + 1e-8) + 
            Math.log(pAttack[discreteOut[2]] + 1e-8)
        );
    }

    public void train(List<StepData> memory) {
        int N = memory.size();
        float[] advantages = new float[N];
        float gae = 0.0f;
        
        for (int i = N - 1; i >= 0; i--) {
            StepData cur = memory.get(i);
            float nextVal = (i == N - 1 || cur.done) ? 0.0f : memory.get(i + 1).value;
            float delta = cur.reward + gamma * nextVal - cur.value;
            gae = delta + gamma * 0.95f * gae;
            advantages[i] = gae;
        }

        for (int epoch = 0; epoch < 4; epoch++) {
            for (int i = 0; i < N; i++) {
                StepData data = memory.get(i);
                float[] hidden = relu(matmulAdd(w1, data.state, b1));
                float[] logits = matmulAdd(wActor, hidden, bActor);
                
                float muPitch = (float) Math.tanh(logits[0]) * 30f;
                float muYaw = (float) Math.tanh(logits[1]) * 40f;
                
                float[] pMove = softmax(slice(logits, 2, 7));
                float[] pJump = softmax(slice(logits, 7, 9));
                float[] pAttack = softmax(slice(logits, 9, 11));

                float curLogProb = -0.5f * (float)(
                    Math.pow((data.pitchDelta - muPitch) / ACTION_STD, 2) + 
                    Math.pow((data.yawDelta - muYaw) / ACTION_STD, 2)
                ) + (float) (
                    Math.log(pMove[data.discreteActions[0]] + 1e-8) + 
                    Math.log(pJump[data.discreteActions[1]] + 1e-8) + 
                    Math.log(pAttack[data.discreteActions[2]] + 1e-8)
                );

                float ratio = (float) Math.exp(curLogProb - data.logProb);
                
                // Backprop (Simplified)
                float clipGrad = (ratio > 1 + clipEps && advantages[i] > 0) || (ratio < 1 - clipEps && advantages[i] < 0) ? 0.0f : -advantages[i];
                for (int h = 0; h < hiddenDim; h++) {
                    for (int s = 0; s < stateDim; s++) {
                        w1.data[h][s] -= lr * clipGrad * data.state[s] * 0.01f;
                    }
                }
            }
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
