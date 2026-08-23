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
                baseAction[i] = Math.max(-1.0f, Math.min(1.0f, baseAction[i] + exploration[i]));
            }
        }
        return baseAction;
    }

    public void trainBatch() {
        if (memory.size() < batchSize) return;

        ContinuousTransition[] batch = memory.sample(batchSize);

        float actorLr = 0.0001f / batchSize;
        float criticLr = 0.001f / batchSize;

        for (ContinuousTransition t : batch) {
            float[] nextAction = targetActor.forward(t.nextState, nextActionScratch);
            float targetQ = targetCritic.evaluate(t.nextState, nextAction);
            float y = t.reward + (t.done ? 0.0f : gamma * targetQ);

            float[] actionGrad = critic.train(t.state, t.action, y, criticLr);

            float[] currentAction = actor.forward(t.state, actionScratch);
            actor.train(t.state, currentAction, actionGrad, actorLr);
        }

        softUpdateTargetNetworks();
    }

    private void softUpdateTargetNetworks() {
        blend(actor.w1, targetActor.w1); blend(actor.b1, targetActor.b1);
        blend(actor.w2, targetActor.w2); blend(actor.b2, targetActor.b2);

        blend(critic.w1, targetCritic.w1); blend(critic.b1, targetCritic.b1);
        blend(critic.w2, targetCritic.w2); blend(critic.b2, targetCritic.b2);
    }

    private void blend(float[] main, float[] target) {
        for (int i = 0; i < main.length; i++) {
            target[i] = tau * main[i] + (1.0f - tau) * target[i];
        }
    }

    public int getMemorySize() {
        return memory.size();
    }
    
    // ADDED: Expose noise reset
    public void resetNoise() {
        noise.reset();
    }
}
