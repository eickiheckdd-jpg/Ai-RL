package com.example.newgen6;

public class DDPGAgent {
    public final ActorNetwork actor, targetActor;
    public final CriticNetwork critic, targetCritic;
    private final ContinuousReplayBuffer memory;
    private final OUNoise noise;
    
    private final int actionSize = 5;
    private final int batchSize = 64;
    private final float gamma = 0.99f;
    private final float tau = 0.005f;
    
    private final float[] actionScratch;
    private final float[] nextActionScratch;

    public DDPGAgent(int stateSize) {
        this.actor = new ActorNetwork(stateSize, actionSize);
        this.targetActor = new ActorNetwork(stateSize, actionSize);
        
        this.critic = new CriticNetwork(stateSize, actionSize);
        this.targetCritic = new CriticNetwork(stateSize, actionSize);
        
        this.memory = new ContinuousReplayBuffer(10000);
        this.noise = new OUNoise(actionSize);
        
        this.actionScratch = new float[actionSize];
        this.nextActionScratch = new float[actionSize];
        
        // Initialize targets
        System.arraycopy(actor.w1, 0, targetActor.w1, 0, actor.w1.length);
        System.arraycopy(critic.w1, 0, targetCritic.w1, 0, critic.w1.length);
    }

    public void addExperience(float[] state, float[] action, float reward, float[] nextState, boolean done) {
        memory.add(state, action, reward, nextState, done);
    }

    public float[] selectAction(float[] state, boolean evaluationMode) {
        float[] baseAction = actor.forward(state, actionScratch);
        
        if (!evaluationMode) {
            float[] exploration = noise.sample();
            for (int i = 0; i < actionSize; i++) {
                // Apply noise and clip to [-1.0, 1.0]
                baseAction[i] = Math.max(-1.0f, Math.min(1.0f, baseAction[i] + exploration[i]));
            }
        }
        return baseAction;
    }

    public void trainBatch() {
        if (memory.size() < batchSize) return;

        ContinuousTransition[] batch = memory.sample(batchSize);

        // Standard DDPG Update Logic:
        // 1. Calculate Target Q: y = r + gamma * Q'(s', \mu'(s'))
        // 2. Update Critic: Minimize MSE(y, Q(s, a))
        // 3. Update Actor: Maximize Q(s, \mu(s)) via Policy Gradient
        
        // TODO: Implement the raw array-based Backpropagation here. 
        // Note: For native Java, writing out the full chain-rule derivatives for DDPG 
        // without a tensor framework is verbose. If performance drops, consider 
        // offloading the actual backprop math to a local Python process via sockets.

        softUpdateTargetNetworks();
    }

    private void softUpdateTargetNetworks() {
        // Soft update Actor
        blend(actor.w1, targetActor.w1); blend(actor.b1, targetActor.b1);
        blend(actor.w2, targetActor.w2); blend(actor.b2, targetActor.b2);

        // Soft update Critic
        blend(critic.w1, targetCritic.w1); blend(critic.b1, targetCritic.b1);
        blend(critic.w2, targetCritic.w2); blend(critic.b2, targetCritic.b2);
    }

    private void blend(float[] main, float[] target) {
        for (int i = 0; i < main.length; i++) {
            target[i] = tau * main[i] + (1.0f - tau) * target[i];
        }
    }
}
