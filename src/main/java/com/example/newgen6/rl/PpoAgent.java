package com.example.newgen6.rl;

import com.example.newgen6.NewGen6RLMod;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

public final class PpoAgent {
    private final Mlp actor;
    private final Mlp critic;

    private final float[] logStd = new float[]{-1.5f, -1.5f};
    private final float[] logStdM = new float[2];
    private final float[] logStdV = new float[2];
    private int logStdT = 0;

    private final RolloutBuffer buffer = new RolloutBuffer(AgentConfig.ROLLOUT_STEPS);
    private final ReentrantLock modelLock = new ReentrantLock();

    private final ExecutorService trainer = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "NEWGEN6-PPO");
        t.setDaemon(true);
        return t;
    });

    private final Random actRng = new Random();
    private final Random trainRng = new Random();

    private final float[] actorGrad = new float[AgentConfig.ACTION_DIM];
    private final float[] criticGrad = new float[1];
    private final float[] logStdGrad = new float[AgentConfig.CONT_ACTIONS];

    public volatile int ppoUpdates = 0;
    public volatile float lastPolicyLoss = 0.0f;
    public volatile float lastValueLoss = 0.0f;
    public volatile float lastEntropy = 0.0f;
    public volatile String checkpointStatus = "none";

    public PpoAgent() {
        Random initRng = new Random(1337);

        actor = new Mlp(
                new int[]{AgentConfig.OBS_DIM, 256, 128, AgentConfig.ACTION_DIM},
                initRng,
                AgentConfig.LEARNING_RATE
        );

        critic = new Mlp(
                new int[]{AgentConfig.OBS_DIM, 256, 128, 1},
                initRng,
                AgentConfig.LEARNING_RATE
        );
    }

    public ActionSample act(float[] obs, boolean deterministic) {
        modelLock.lock();
        try {
            float[] actorOut = actor.forward(obs);
            float value = critic.forward(obs)[0];

            boolean[] keys = new boolean[AgentConfig.KEY_ACTIONS];
            float[] a = new float[AgentConfig.ACTION_DIM];

            for (int k = 0; k < AgentConfig.KEY_ACTIONS; k++) {
                float p = sigmoid(actorOut[k]);

                if (deterministic) {
                    keys[k] = p >= 0.5f;
                } else {
                    keys[k] = actRng.nextFloat() < p;
                }

                a[k] = keys[k] ? 1.0f : 0.0f;
            }

            float mouseX;
            float mouseY;

            if (deterministic) {
                mouseX = actorOut[8];
                mouseY = actorOut[9];
            } else {
                float stdX = (float) Math.exp(clamp(logStd[0], -5.0f, 2.0f));
                float stdY = (float) Math.exp(clamp(logStd[1], -5.0f, 2.0f));

                mouseX = (float) (actorOut[8] + stdX * actRng.nextGaussian());
                mouseY = (float) (actorOut[9] + stdY * actRng.nextGaussian());
            }

            mouseX = clamp(mouseX, -AgentConfig.MOUSE_CLAMP, AgentConfig.MOUSE_CLAMP);
            mouseY = clamp(mouseY, -AgentConfig.MOUSE_CLAMP, AgentConfig.MOUSE_CLAMP);

            a[8] = mouseX;
            a[9] = mouseY;

            float logProb = computeLogProb(actorOut, logStd, a, 0);

            return new ActionSample(keys, mouseX, mouseY, logProb, value);
        } finally {
            modelLock.unlock();
        }
    }

    public void store(float[] obs, ActionSample action, float reward, boolean done) {
        if (buffer.isFull()) {
            buffer.reset();
        }

        buffer.add(obs, action.vector, action.logProb, action.value, reward, done);

        if (buffer.isFull()) {
            float lastValue = done ? 0.0f : evaluateValue(obs);
            PpoBatch batch = buffer.createBatch(lastValue);
            buffer.reset();
            submitTraining(batch);
        }
    }

    private float evaluateValue(float[] obs) {
        modelLock.lock();
        try {
            return critic.forward(obs)[0];
        } finally {
            modelLock.unlock();
        }
    }

    private void submitTraining(PpoBatch batch) {
        trainer.submit(() -> {
            try {
                train(batch);
            } catch (Throwable t) {
                NewGen6RLMod.LOGGER.error("[NEWGEN6] PPO training error", t);
            }
        });
    }

    private void train(PpoBatch batch) {
        batch.computeAdvantages(AgentConfig.GAMMA, AgentConfig.LAMBDA);

        Mlp trainActor;
        Mlp trainCritic;
        float[] trainLogStd;

        modelLock.lock();
        try {
            trainActor = new Mlp(actor);
            trainCritic = new Mlp(critic);
            trainLogStd = logStd.clone();
        } finally {
            modelLock.unlock();
        }

        int[] indices = new int[batch.count];
        for (int i = 0; i < batch.count; i++) indices[i] = i;

        for (int epoch = 0; epoch < AgentConfig.PPO_EPOCHS; epoch++) {
            shuffle(indices);

            for (int start = 0; start < batch.count; start += AgentConfig.MINIBATCH_SIZE) {
                int end = Math.min(start + AgentConfig.MINIBATCH_SIZE, batch.count);
                int n = end - start;

                trainActor.zeroGrad();
                trainCritic.zeroGrad();
                Arrays.fill(logStdGrad, 0.0f);

                float entropySum = 0.0f;
                float policyLossSum = 0.0f;
                float valueLossSum = 0.0f;
                boolean bad = false;

                for (int s = start; s < end; s++) {
                    int idx = indices[s];

                    int obsOff = idx * AgentConfig.OBS_DIM;
                    int actOff = idx * AgentConfig.ACTION_DIM;

                    float[] out = trainActor.forward(batch.obs, obsOff);
                    float newLogProb = computeLogProb(out, trainLogStd, batch.actions, actOff);
                    float oldLogProb = batch.oldLogProbs[idx];

                    if (!Float.isFinite(newLogProb)) {
                        bad = true;
                        break;
                    }

                    float ratio = (float) Math.exp(clamp(newLogProb - oldLogProb, -20.0f, 20.0f));
                    float adv = batch.advantages[idx];

                    float raw = ratio * adv;
                    float clipped = clamp(ratio, 1.0f - AgentConfig.CLIP_EPS, 1.0f + AgentConfig.CLIP_EPS) * adv;

                    boolean clippedBranch =
                            (adv > 0.0f && ratio > 1.0f + AgentConfig.CLIP_EPS) ||
                            (adv < 0.0f && ratio < 1.0f - AgentConfig.CLIP_EPS);

                    float gLogProb = clippedBranch ? 0.0f : -adv * ratio;

                    Arrays.fill(actorGrad, 0.0f);

                    // Discrete Bernoulli actions
                    for (int k = 0; k < AgentConfig.KEY_ACTIONS; k++) {
                        float logit = out[k];
                        float p = sigmoid(logit);
                        float y = batch.actions[actOff + k];

                        actorGrad[k] = gLogProb * (y - p)
                                + AgentConfig.ENTROPY_COEF * logit * p * (1.0f - p);

                        entropySum += bernoulliEntropy(p);
                    }

                    // Continuous mouse deltas
                    for (int j = 0; j < AgentConfig.CONT_ACTIONS; j++) {
                        int outIdx = AgentConfig.KEY_ACTIONS + j;

                        float mean = out[outIdx];
                        float y = batch.actions[actOff + outIdx];
                        float ls = trainLogStd[j];

                        float std = (float) Math.exp(clamp(ls, -5.0f, 2.0f));
                        float var = std * std;
                        float diff = y - mean;

                        actorGrad[outIdx] = gLogProb * diff / var;

                        logStdGrad[j] += gLogProb * ((diff * diff / var) - 1.0f)
                                - AgentConfig.ENTROPY_COEF;

                        entropySum += 0.5f * (float) Math.log(2.0f * Math.PI * Math.E * var);
                    }

                    trainActor.backward(actorGrad);

                    float v = trainCritic.forward(batch.obs, obsOff)[0];
                    float ret = batch.returns[idx];
                    float vDiff = v - ret;

                    criticGrad[0] = vDiff;
                    trainCritic.backward(criticGrad);

                    policyLossSum += -Math.min(raw, clipped);
                    valueLossSum += 0.5f * vDiff * vDiff;
                }

                if (bad || !trainActor.isFinite() || !trainCritic.isFinite()) {
                    NewGen6RLMod.LOGGER.warn("[NEWGEN6] NaN/Inf detected during PPO update; skipping apply");
                    return;
                }

                float scale = 1.0f / Math.max(1, n);
                trainActor.scaleGrad(scale);
                trainCritic.scaleGrad(scale);

                trainActor.step(1.0f);
                trainCritic.step(1.0f);

                updateLogStd(trainLogStd, scale);

                lastPolicyLoss = policyLossSum / n;
                lastValueLoss = valueLossSum / n;
                lastEntropy = entropySum / n;
            }
        }

        if (trainActor.isFinite() && trainCritic.isFinite()) {
            modelLock.lock();
            try {
                actor.copyFrom(trainActor);
                critic.copyFrom(trainCritic);
                System.arraycopy(trainLogStd, 0, logStd, 0, logStd.length);
                ppoUpdates++;
            } finally {
                modelLock.unlock();
            }
        }
    }

    private void updateLogStd(float[] trainLogStd, float scale) {
        logStdT++;

        float beta1Pow = (float) Math.pow(0.9, logStdT);
        float beta2Pow = (float) Math.pow(0.999, logStdT);

        float corr1 = 1.0f - beta1Pow;
        float corr2 = 1.0f - beta2Pow;

        for (int i = 0; i < trainLogStd.length; i++) {
            float g = logStdGrad[i] * scale;
            if (!Float.isFinite(g)) g = 0.0f;
            g = clamp(g, -1.0f, 1.0f);

            logStdM[i] = 0.9f * logStdM[i] + 0.1f * g;
            logStdV[i] = 0.999f * logStdV[i] + 0.001f * g * g;

            float mHat = logStdM[i] / corr1;
            float vHat = logStdV[i] / corr2;

            trainLogStd[i] -= AgentConfig.LEARNING_RATE * mHat / ((float) Math.sqrt(vHat) + 1e-5f);
            trainLogStd[i] = clamp(trainLogStd[i], -5.0f, 2.0f);
        }
    }

    private void shuffle(int[] arr) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = trainRng.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    private float computeLogProb(float[] out, float[] ls, float[] actions, int off) {
        float lp = 0.0f;

        for (int k = 0; k < AgentConfig.KEY_ACTIONS; k++) {
            float p = sigmoid(out[k]);
            p = clamp(p, 1e-6f, 1.0f - 1e-6f);

            float y = actions[off + k];
            lp += y * (float) Math.log(p) + (1.0f - y) * (float) Math.log(1.0f - p);
        }

        for (int j = 0; j < AgentConfig.CONT_ACTIONS; j++) {
            int idx = AgentConfig.KEY_ACTIONS + j;

            float mean = out[idx];
            float x = actions[off + idx];
            float lstd = clamp(ls[j], -5.0f, 2.0f);
            float std = (float) Math.exp(lstd);
            float diff = x - mean;

            lp += -0.5f * (
                    (diff * diff) / (std * std)
                            + 2.0f * lstd
                            + (float) Math.log(2.0f * Math.PI)
            );
        }

        return lp;
    }

    private static float sigmoid(float x) {
        x = clamp(x, -20.0f, 20.0f);
        return (float) (1.0f / (1.0f + Math.exp(-x)));
    }

    private static float bernoulliEntropy(float p) {
        if (p <= 1e-6f || p >= 1.0f - 1e-6f) return 0.0f;
        return (float) (-(p * Math.log(p) + (1.0f - p) * Math.log(1.0f - p)));
    }

    private static float clamp(float v, float min, float max) {
        if (!Float.isFinite(v)) return min;
        return Math.max(min, Math.min(max, v));
    }

    public void saveCheckpointAsync(Path path, int episode, long globalStep, Runnable onDone) {
        trainer.submit(() -> {
            try {
                ModelCheckpoint.save(this, path, episode, globalStep);
                checkpointStatus = "saved";
            } catch (Throwable t) {
                checkpointStatus = "error";
                NewGen6RLMod.LOGGER.error("[NEWGEN6] Failed saving checkpoint", t);
            } finally {
                if (onDone != null) {
                    onDone.run();
                }
            }
        });
    }

    public boolean loadCheckpoint(Path path) {
        checkpointStatus = "loading";
        boolean ok = ModelCheckpoint.load(this, path);
        checkpointStatus = ok ? "loaded" : "none";
        return ok;
    }

    public void writeModel(DataOutputStream out) throws IOException {
        modelLock.lock();
        try {
            out.writeInt(logStd.length);

            for (float v : logStd) out.writeFloat(v);
            for (float v : logStdM) out.writeFloat(v);
            for (float v : logStdV) out.writeFloat(v);

            out.writeInt(logStdT);
            out.writeInt(ppoUpdates);
            out.writeFloat(lastPolicyLoss);
            out.writeFloat(lastValueLoss);
            out.writeFloat(lastEntropy);

            actor.write(out);
            critic.write(out);
        } finally {
            modelLock.unlock();
        }
    }

    public boolean readModel(DataInputStream in) {
        try {
            int logLen = in.readInt();
            if (logLen != AgentConfig.CONT_ACTIONS) return false;

            float[] newLogStd = new float[logLen];
            float[] newLogStdM = new float[logLen];
            float[] newLogStdV = new float[logLen];

            for (int i = 0; i < logLen; i++) newLogStd[i] = in.readFloat();
            for (int i = 0; i < logLen; i++) newLogStdM[i] = in.readFloat();
            for (int i = 0; i < logLen; i++) newLogStdV[i] = in.readFloat();

            int newLogStdT = in.readInt();
            int newPpoUpdates = in.readInt();
            float newPolicyLoss = in.readFloat();
            float newValueLoss = in.readFloat();
            float newEntropy = in.readFloat();

            Mlp newActor = Mlp.read(in);
            Mlp newCritic = Mlp.read(in);

            if (!validTopology(newActor, newCritic)) {
                return false;
            }

            modelLock.lock();
            try {
                actor.copyFrom(newActor);
                critic.copyFrom(newCritic);

                System.arraycopy(newLogStd, 0, logStd, 0, logStd.length);
                System.arraycopy(newLogStdM, 0, logStdM, 0, logStdM.length);
                System.arraycopy(newLogStdV, 0, logStdV, 0, logStdV.length);

                logStdT = newLogStdT;
                ppoUpdates = newPpoUpdates;
                lastPolicyLoss = newPolicyLoss;
                lastValueLoss = newValueLoss;
                lastEntropy = newEntropy;
                checkpointStatus = "loaded";
            } finally {
                modelLock.unlock();
            }

            return true;
        } catch (Exception e) {
            NewGen6RLMod.LOGGER.error("[NEWGEN6] Failed reading model checkpoint", e);
            return false;
        }
    }

    private boolean validTopology(Mlp a, Mlp c) {
        return a.sizes[0] == AgentConfig.OBS_DIM
                && a.sizes[a.sizes.length - 1] == AgentConfig.ACTION_DIM
                && c.sizes[0] == AgentConfig.OBS_DIM
                && c.sizes[c.sizes.length - 1] == 1;
    }
}