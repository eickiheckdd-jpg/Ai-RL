package com.example.newgen6.rl.ppo;

/**
 * PPO hyperparameters. Defaults below are reasonable starting points for a
 * small on-policy discrete-action network with a 2048-tick rollout; final
 * tuning is the LLM Architect's call.
 */
public class PPOConfig {
    public float gamma = 0.99f;
    public float lambda = 0.95f;
    public float clipEps = 0.2f;
    public float entropyCoef = 0.01f;
    public float valueCoef = 0.5f;
    public float learningRate = 3e-4f;
    public int epochs = 4;
    public int minibatchSize = 64;
    public int bufferSize = 2048;
}
