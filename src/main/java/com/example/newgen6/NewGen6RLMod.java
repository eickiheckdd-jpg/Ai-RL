package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.player.PlayerEntity;
import org.lwjgl.glfw.GLFW;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public final class NewGen6RLMod implements ClientModInitializer {

    public static final Logger LOGGER = LoggerFactory.getLogger("newgen6");

    private static final int OBSERVATION_SIZE = 12;
    private static final int HIDDEN_SIZE = 64;
    private static final int ROLLOUT_SIZE = 512;
    private static final int PPO_EPOCHS = 4;

    private static final float GAMMA = 0.99f;
    private static final float GAE_LAMBDA = 0.95f;
    private static final float CLIP_RANGE = 0.20f;
    private static final float POLICY_LEARNING_RATE = 0.0003f;
    private static final float VALUE_COEFFICIENT = 0.5f;
    private static final float ENTROPY_COEFFICIENT = 0.01f;
    private static final float MAX_GRAD_NORM = 0.5f;

    private static final float MAX_YAW_DELTA_DEGREES = 2.5f;
    private static final float MAX_PITCH_DELTA_DEGREES = 2.0f;
    private static final double TARGET_RANGE = 32.0;

    private static final KeyBinding TOGGLE_AI = KeyBindingHelper.registerKeyBinding(
            new KeyBinding(
                    "key.newgen6.toggle_ai",
                    InputUtil.Type.KEYSYM,
                    GLFW.GLFW_KEY_C,
                    KeyBinding.Category.MISC
            )
    );

    private static final AimNetwork NETWORK = new AimNetwork(
            OBSERVATION_SIZE, HIDDEN_SIZE, new Random()
    );

    private static final RolloutBuffer ROLLOUT = new RolloutBuffer(ROLLOUT_SIZE);
    private static final Random RNG = new Random();

    private static boolean aiEnabled = false;
    private static float previousAimError = Float.NaN;
    private static long environmentSteps = 0;
    private static long ppoUpdates = 0;
    private static long episodes = 0;
    private static double episodeReward = 0.0;

    @Override
    public void onInitializeClient() {
        LOGGER.info("NEWGEN6 RL client initialized.");
        LOGGER.info("Target Minecraft: 1.21.11");
        LOGGER.info("AI starts OFF. Press C to toggle.");
        ClientTickEvents.END_CLIENT_TICK.register(NewGen6RLMod::onClientTick);
    }

    private static void onClientTick(MinecraftClient client) {
        while (TOGGLE_AI.wasPressed()) {
            aiEnabled = !aiEnabled;
            if (!aiEnabled) {
                resetEpisodeState();
            }
            LOGGER.info("NEWGEN6 AI: {}", aiEnabled ? "ON" : "OFF");
        }

        if (!aiEnabled || client.player == null || client.world == null) {
            return;
        }

        try {
            step(client);
        } catch (Throwable throwable) {
            aiEnabled = false;
            resetEpisodeState();
            LOGGER.error("NEWGEN6 AI disabled after runtime error.", throwable);
        }
    }

    private static void step(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        PlayerEntity target = findNearestTarget(player);

        if (target == null) {
            if (!Float.isNaN(previousAimError)) {
                finishEpisode();
            }
            previousAimError = Float.NaN;
            return;
        }

        Observation observation = Observation.from(player, target);
        if (!observation.isFinite()) {
            return;
        }

        AimNetwork.Action action = NETWORK.sampleAction(observation.values, RNG);
        float oldError = observation.angularError();

        applyCameraAction(player, action);

        Observation nextObservation = Observation.from(player, target);
        if (!nextObservation.isFinite()) {
            return;
        }

        float newError = nextObservation.angularError();
        float baselineError = Float.isNaN(previousAimError) ? oldError : previousAimError;
        float reward = clamp(baselineError - newError - 0.001f, -1.0f, 1.0f);

        previousAimError = newError;
        episodeReward += reward;
        environmentSteps++;

        ROLLOUT.add(
                observation.values,
                action.yaw,
                action.pitch,
                action.logProbability,
                action.value,
                reward
        );

        if (ROLLOUT.isFull()) {
            trainPPO();
            ROLLOUT.clear();
        }

        if (environmentSteps % 100 == 0) {
            LOGGER.info(
                    "NEWGEN6 | steps={} updates={} episodes={} reward={} yawErr={} pitchErr={} entropy={}",
                    environmentSteps,
                    ppoUpdates,
                    episodes,
                    round(episodeReward),
                    round(nextObservation.yawError),
                    round(nextObservation.pitchError),
                    round(NETWORK.lastEntropy)
            );
        }
    }

    private static void applyCameraAction(ClientPlayerEntity player, AimNetwork.Action action) {
        float newYaw = player.getYaw() + action.yaw * MAX_YAW_DELTA_DEGREES;
        float newPitch = clamp(
                player.getPitch() + action.pitch * MAX_PITCH_DELTA_DEGREES,
                -90.0f,
                90.0f
        );
        player.setYaw(newYaw);
        player.setPitch(newPitch);
    }

    private static PlayerEntity findNearestTarget(ClientPlayerEntity self) {
        if (self.getWorld() == null) {
            return null;
        }

        double bestDistanceSquared = TARGET_RANGE * TARGET_RANGE;
        PlayerEntity best = null;

        for (PlayerEntity candidate : self.getWorld().getPlayers()) {
            if (candidate == self || candidate.isSpectator() || !candidate.isAlive()) {
                continue;
            }

            double dx = candidate.getX() - self.getX();
            double dy = candidate.getY() - self.getY();
            double dz = candidate.getZ() - self.getZ();
            double distanceSquared = dx * dx + dy * dy + dz * dz;

            if (distanceSquared < bestDistanceSquared) {
                bestDistanceSquared = distanceSquared;
                best = candidate;
            }
        }

        return best;
    }

    private static void trainPPO() {
        if (ROLLOUT.size < 8) {
            return;
        }

        ROLLOUT.finishAdvantages();

        float totalPolicyLoss = 0.0f;
        float totalValueLoss = 0.0f;
        float totalEntropy = 0.0f;

        for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            for (int i = 0; i < ROLLOUT.size; i++) {
                AimNetwork.TrainingResult result = NETWORK.trainSample(
                        ROLLOUT.observations[i],
                        ROLLOUT.yawActions[i],
                        ROLLOUT.pitchActions[i],
                        ROLLOUT.oldLogProbabilities[i],
                        ROLLOUT.advantages[i],
                        ROLLOUT.returns[i]
                );

                totalPolicyLoss += result.policyLoss;
                totalValueLoss += result.valueLoss;
                totalEntropy += result.entropy;
            }

            NETWORK.applyAdam(POLICY_LEARNING_RATE, MAX_GRAD_NORM);
        }

        ppoUpdates++;
        float sampleCount = ROLLOUT.size * PPO_EPOCHS;

        LOGGER.info(
                "NEWGEN6 PPO UPDATE #{} | policyLoss={} valueLoss={} entropy={}",
                ppoUpdates,
                round(totalPolicyLoss / sampleCount),
                round(totalValueLoss / sampleCount),
                round(totalEntropy / sampleCount)
        );
    }

    private static void finishEpisode() {
        episodes++;
        LOGGER.info(
                "NEWGEN6 EPISODE #{} | reward={}",
                episodes,
                round(episodeReward)
        );
        episodeReward = 0.0;
        previousAimError = Float.NaN;
    }

    private static void resetEpisodeState() {
        previousAimError = Float.NaN;
        episodeReward = 0.0;
        ROLLOUT.clear();
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double round(double value) {
        return Math.rint(value * 10000.0) / 10000.0;
    }

    private static float wrapDegrees(float angle) {
        float wrapped = angle % 360.0f;
        if (wrapped >= 180.0f) wrapped -= 360.0f;
        if (wrapped < -180.0f) wrapped += 360.0f;
        return wrapped;
    }

    public static final class Observation {
        public final float[] values;
        public final float distance;
        public final float yawError;
        public final float pitchError;

        private Observation(float[] values, float distance, float yawError, float pitchError) {
            this.values = values;
            this.distance = distance;
            this.yawError = yawError;
            this.pitchError = pitchError;
        }

        public static Observation from(ClientPlayerEntity player, PlayerEntity target) {
            double selfX = player.getX();
            double selfY = player.getY() + 1.0;
            double selfZ = player.getZ();

            double targetX = target.getX();
            double targetY = target.getY() + 1.0;
            double targetZ = target.getZ();

            float relativeX = (float) (targetX - selfX);
            float relativeY = (float) (targetY - selfY);
            float relativeZ = (float) (targetZ - selfZ);

            float horizontalDistance = (float) Math.sqrt(
                    relativeX * relativeX + relativeZ * relativeZ
            );
            float distance = (float) Math.sqrt(
                    relativeX * relativeX
                            + relativeY * relativeY
                            + relativeZ * relativeZ
            );

            float targetYaw = (float) Math.toDegrees(
                    Math.atan2(-relativeX, -relativeZ)
            );
            float targetPitch = (float) Math.toDegrees(
                    -Math.atan2(relativeY, horizontalDistance)
            );

            float yawError = wrapDegrees(targetYaw - player.getYaw());
            float pitchError = targetPitch - player.getPitch();

            var selfVelocity = player.getVelocity();
            var targetVelocity = target.getVelocity();

            float[] values = new float[] {
                    clamp(relativeX / 20.0f, -4.0f, 4.0f),
                    clamp(relativeY / 20.0f, -4.0f, 4.0f),
                    clamp(relativeZ / 20.0f, -4.0f, 4.0f),
                    clamp(distance / 20.0f, 0.0f, 4.0f),
                    yawError / 180.0f,
                    pitchError / 180.0f,
                    clamp((float) selfVelocity.x / 5.0f, -2.0f, 2.0f),
                    clamp((float) selfVelocity.y / 5.0f, -2.0f, 2.0f),
                    clamp((float) selfVelocity.z / 5.0f, -2.0f, 2.0f),
                    clamp((float) targetVelocity.x / 5.0f, -2.0f, 2.0f),
                    clamp((float) targetVelocity.y / 5.0f, -2.0f, 2.0f),
                    clamp((float) targetVelocity.z / 5.0f, -2.0f, 2.0f)
            };

            return new Observation(values, distance, yawError, pitchError);
        }

        public float angularError() {
            return (float) Math.sqrt(
                    yawError * yawError + pitchError * pitchError
            ) / 180.0f;
        }

        public boolean isFinite() {
            if (!Float.isFinite(distance)
                    || !Float.isFinite(yawError)
                    || !Float.isFinite(pitchError)) {
                return false;
            }
            for (float value : values) {
                if (!Float.isFinite(value)) return false;
            }
            return true;
        }
    }

    public static final class RolloutBuffer {
        private final int capacity;
        private int size = 0;

        private final float[][] observations;
        private final float[] yawActions;
        private final float[] pitchActions;
        private final float[] oldLogProbabilities;
        private final float[] oldValues;
        private final float[] rewards;
        private final float[] advantages;
        private final float[] returns;

        RolloutBuffer(int capacity) {
            this.capacity = capacity;
            observations = new float[capacity][OBSERVATION_SIZE];
            yawActions = new float[capacity];
            pitchActions = new float[capacity];
            oldLogProbabilities = new float[capacity];
            oldValues = new float[capacity];
            rewards = new float[capacity];
            advantages = new float[capacity];
            returns = new float[capacity];
        }

        void add(
                float[] observation,
                float yawAction,
                float pitchAction,
                float oldLogProbability,
                float oldValue,
                float reward
        ) {
            if (size >= capacity) return;

            System.arraycopy(observation, 0, observations[size], 0, OBSERVATION_SIZE);
            yawActions[size] = yawAction;
            pitchActions[size] = pitchAction;
            oldLogProbabilities[size] = oldLogProbability;
            oldValues[size] = oldValue;
            rewards[size] = reward;
            size++;
        }

        boolean isFull() {
            return size >= capacity;
        }

        void clear() {
            size = 0;
        }

        void finishAdvantages() {
            float nextValue = 0.0f;
            float runningAdvantage = 0.0f;

            for (int i = size - 1; i >= 0; i--) {
                float delta = rewards[i] + GAMMA * nextValue - oldValues[i];
                runningAdvantage = delta
                        + GAMMA * GAE_LAMBDA * runningAdvantage;

                advantages[i] = runningAdvantage;
                returns[i] = advantages[i] + oldValues[i];
                nextValue = oldValues[i];
            }

            double mean = 0.0;
            for (int i = 0; i < size; i++) mean += advantages[i];
            mean /= size;

            double variance = 0.0;
            for (int i = 0; i < size; i++) {
                double delta = advantages[i] - mean;
                variance += delta * delta;
            }
            variance /= size;

            double std = Math.sqrt(variance + 1e-8);
            for (int i = 0; i < size; i++) {
                advantages[i] = (float) ((advantages[i] - mean) / std);
            }
        }
    }

    public static final class AimNetwork {
        private final int inputSize;
        private final int hiddenSize;

        private final float[][] inputWeights;
        private final float[] hiddenBias;
        private final float[][] outputWeights;
        private final float[] outputBias;

        private final float[][] inputWeightGradients;
        private final float[] hiddenBiasGradients;
        private final float[][] outputWeightGradients;
        private final float[] outputBiasGradients;

        private final float[][] inputWeightAdamM;
        private final float[][] inputWeightAdamV;
        private final float[] hiddenBiasAdamM;
        private final float[] hiddenBiasAdamV;
        private final float[][] outputWeightAdamM;
        private final float[][] outputWeightAdamV;
        private final float[] outputBiasAdamM;
        private final float[] outputBiasAdamV;

        private long adamStep = 0;
        public float lastEntropy = 0.0f;

        AimNetwork(int inputSize, int hiddenSize, Random random) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;

            inputWeights = new float[inputSize][hiddenSize];
            hiddenBias = new float[hiddenSize];
            outputWeights = new float[hiddenSize][5];
            outputBias = new float[5];

            inputWeightGradients = new float[inputSize][hiddenSize];
            hiddenBiasGradients = new float[hiddenSize];
            outputWeightGradients = new float[hiddenSize][5];
            outputBiasGradients = new float[5];

            inputWeightAdamM = new float[inputSize][hiddenSize];
            inputWeightAdamV = new float[inputSize][hiddenSize];
            hiddenBiasAdamM = new float[hiddenSize];
            hiddenBiasAdamV = new float[hiddenSize];
            outputWeightAdamM = new float[hiddenSize][5];
            outputWeightAdamV = new float[hiddenSize][5];
            outputBiasAdamM = new float[5];
            outputBiasAdamV = new float[5];

            initialize(random);
        }

        private void initialize(Random random) {
            float inputScale = (float) Math.sqrt(2.0 / inputSize);
            float hiddenScale = (float) Math.sqrt(2.0 / hiddenSize);

            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    inputWeights[i][j] =
                            (random.nextFloat() * 2.0f - 1.0f) * inputScale;
                }
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < 5; j++) {
                    outputWeights[i][j] =
                            (random.nextFloat() * 2.0f - 1.0f) * hiddenScale;
                }
            }

            outputBias[2] = -0.5f;
            outputBias[3] = -0.5f;
        }

        Action sampleAction(float[] observation, Random random) {
            ForwardPass pass = forward(observation);

            float yawLogStd = clamp(pass.output[2], -4.0f, 1.0f);
            float pitchLogStd = clamp(pass.output[3], -4.0f, 1.0f);

            float yawStd = (float) Math.exp(yawLogStd);
            float pitchStd = (float) Math.exp(pitchLogStd);

            float yawZ = pass.output[0] + yawStd * gaussian(random);
            float pitchZ = pass.output[1] + pitchStd * gaussian(random);

            float yaw = (float) Math.tanh(yawZ);
            float pitch = (float) Math.tanh(pitchZ);

            float logProbability =
                    squashedGaussianLogProbability(yaw, yawZ, pass.output[0], yawLogStd)
                            + squashedGaussianLogProbability(
                            pitch, pitchZ, pass.output[1], pitchLogStd
                    );

            float entropy = 0.5f * (
                    (float) Math.log(2.0 * Math.PI * Math.E) + 2.0f * yawLogStd
                            + (float) Math.log(2.0 * Math.PI * Math.E) + 2.0f * pitchLogStd
            );

            lastEntropy = entropy;
            return new Action(yaw, pitch, logProbability, pass.output[4]);
        }

        TrainingResult trainSample(
                float[] observation,
                float yawAction,
                float pitchAction,
                float oldLogProbability,
                float advantage,
                float targetValue
        ) {
            ForwardPass pass = forward(observation);

            float yawLogStd = clamp(pass.output[2], -4.0f, 1.0f);
            float pitchLogStd = clamp(pass.output[3], -4.0f, 1.0f);

            float yaw = clamp(yawAction, -0.999999f, 0.999999f);
            float pitch = clamp(pitchAction, -0.999999f, 0.999999f);

            float yawZ = atanh(yaw);
            float pitchZ = atanh(pitch);

            float currentLogProbability =
                    squashedGaussianLogProbability(yaw, yawZ, pass.output[0], yawLogStd)
                            + squashedGaussianLogProbability(
                            pitch, pitchZ, pass.output[1], pitchLogStd
                    );

            float logRatio = clamp(
                    currentLogProbability - oldLogProbability,
                    -8.0f,
                    8.0f
            );
            float ratio = (float) Math.exp(logRatio);

            boolean clipped =
                    (advantage > 0.0f && ratio > 1.0f + CLIP_RANGE)
                            || (advantage < 0.0f && ratio < 1.0f - CLIP_RANGE);

            float policyCoefficient = clipped ? 0.0f : ratio * advantage;

            float yawVariance = (float) Math.exp(2.0f * yawLogStd);
            float pitchVariance = (float) Math.exp(2.0f * pitchLogStd);

            float[] outputGradient = new float[5];
            outputGradient[0] = policyCoefficient
                    * (yawZ - pass.output[0]) / yawVariance;
            outputGradient[1] = policyCoefficient
                    * (pitchZ - pass.output[1]) / pitchVariance;
            outputGradient[2] = policyCoefficient
                    * (((yawZ - pass.output[0]) * (yawZ - pass.output[0])) / yawVariance - 1.0f)
                    + ENTROPY_COEFFICIENT;
            outputGradient[3] = policyCoefficient
                    * (((pitchZ - pass.output[1]) * (pitchZ - pass.output[1])) / pitchVariance - 1.0f)
                    + ENTROPY_COEFFICIENT;
            outputGradient[4] = -VALUE_COEFFICIENT * (pass.output[4] - targetValue);

            for (int i = 0; i < outputGradient.length; i++) {
                outputGradient[i] = clamp(outputGradient[i], -10.0f, 10.0f);
            }

            for (int h = 0; h < hiddenSize; h++) {
                for (int o = 0; o < 5; o++) {
                    outputWeightGradients[h][o] += pass.hidden[h] * outputGradient[o];
                }
            }

            for (int o = 0; o < 5; o++) {
                outputBiasGradients[o] += outputGradient[o];
            }

            float[] hiddenGradient = new float[hiddenSize];

            for (int h = 0; h < hiddenSize; h++) {
                float value = 0.0f;
                for (int o = 0; o < 5; o++) {
                    value += outputWeights[h][o] * outputGradient[o];
                }
                hiddenGradient[h] = pass.preActivation[h] > 0.0f ? value : 0.0f;
            }

            for (int i = 0; i < inputSize; i++) {
                for (int h = 0; h < hiddenSize; h++) {
                    inputWeightGradients[i][h] += observation[i] * hiddenGradient[h];
                }
            }

            for (int h = 0; h < hiddenSize; h++) {
                hiddenBiasGradients[h] += hiddenGradient[h];
            }

            float policyLoss = clipped ? -Math.abs(advantage) : -(ratio * advantage);
            float valueError = pass.output[4] - targetValue;
            float valueLoss = 0.5f * valueError * valueError;
            float entropy = 0.5f * (
                    (float) Math.log(2.0 * Math.PI * Math.E) + 2.0f * yawLogStd
                            + (float) Math.log(2.0 * Math.PI * Math.E) + 2.0f * pitchLogStd
            );

            lastEntropy = entropy;
            return new TrainingResult(policyLoss, valueLoss, entropy);
        }

        ForwardPass forward(float[] observation) {
            float[] preActivation = new float[hiddenSize];
            float[] hidden = new float[hiddenSize];
            float[] output = new float[5];

            for (int h = 0; h < hiddenSize; h++) {
                float sum = hiddenBias[h];
                for (int i = 0; i < inputSize; i++) {
                    sum += observation[i] * inputWeights[i][h];
                }
                preActivation[h] = sum;
                hidden[h] = Math.max(0.0f, sum);
            }

            for (int o = 0; o < 5; o++) {
                float sum = outputBias[o];
                for (int h = 0; h < hiddenSize; h++) {
                    sum += hidden[h] * outputWeights[h][o];
                }
                output[o] = sum;
            }

            return new ForwardPass(preActivation, hidden, output);
        }

        void applyAdam(float learningRate, float maxGradNorm) {
            adamStep++;

            float totalSquaredGradient = 0.0f;

            for (int i = 0; i < inputSize; i++) {
                for (int h = 0; h < hiddenSize; h++) {
                    totalSquaredGradient += inputWeightGradients[i][h] * inputWeightGradients[i][h];
                }
            }

            for (int h = 0; h < hiddenSize; h++) {
                totalSquaredGradient += hiddenBiasGradients[h] * hiddenBiasGradients[h];
            }

            for (int h = 0; h < hiddenSize; h++) {
                for (int o = 0; o < 5; o++) {
                    totalSquaredGradient += outputWeightGradients[h][o] * outputWeightGradients[h][o];
                }
            }

            for (int o = 0; o < 5; o++) {
                totalSquaredGradient += outputBiasGradients[o] * outputBiasGradients[o];
            }

            float norm = (float) Math.sqrt(totalSquaredGradient + 1e-8);
            float scale = norm > maxGradNorm ? maxGradNorm / norm : 1.0f;

            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float epsilon = 1e-8f;
            float correction1 = 1.0f - (float) Math.pow(beta1, adamStep);
            float correction2 = 1.0f - (float) Math.pow(beta2, adamStep);

            for (int i = 0; i < inputSize; i++) {
                for (int h = 0; h < hiddenSize; h++) {
                    float g = inputWeightGradients[i][h] * scale;
                    inputWeightAdamM[i][h] = beta1 * inputWeightAdamM[i][h] + (1.0f - beta1) * g;
                    inputWeightAdamV[i][h] = beta2 * inputWeightAdamV[i][h] + (1.0f - beta2) * g * g;
                    inputWeights[i][h] += learningRate
                            * (inputWeightAdamM[i][h] / correction1)
                            / ((float) Math.sqrt(inputWeightAdamV[i][h] / correction2) + epsilon);
                    inputWeightGradients[i][h] = 0.0f;
                }
            }

            for (int h = 0; h < hiddenSize; h++) {
                float g = hiddenBiasGradients[h] * scale;
                hiddenBiasAdamM[h] = beta1 * hiddenBiasAdamM[h] + (1.0f - beta1) * g;
                hiddenBiasAdamV[h] = beta2 * hiddenBiasAdamV[h] + (1.0f - beta2) * g * g;
                hiddenBias[h] += learningRate
                        * (hiddenBiasAdamM[h] / correction1)
                        / ((float) Math.sqrt(hiddenBiasAdamV[h] / correction2) + epsilon);
                hiddenBiasGradients[h] = 0.0f;
            }

            for (int h = 0; h < hiddenSize; h++) {
                for (int o = 0; o < 5; o++) {
                    float g = outputWeightGradients[h][o] * scale;
                    outputWeightAdamM[h][o] = beta1 * outputWeightAdamM[h][o] + (1.0f - beta1) * g;
                    outputWeightAdamV[h][o] = beta2 * outputWeightAdamV[h][o] + (1.0f - beta2) * g * g;
                    outputWeights[h][o] += learningRate
                            * (outputWeightAdamM[h][o] / correction1)
                            / ((float) Math.sqrt(outputWeightAdamV[h][o] / correction2) + epsilon);
                    outputWeightGradients[h][o] = 0.0f;
                }
            }

            for (int o = 0; o < 5; o++) {
                float g = outputBiasGradients[o] * scale;
                outputBiasAdamM[o] = beta1 * outputBiasAdamM[o] + (1.0f - beta1) * g;
                outputBiasAdamV[o] = beta2 * outputBiasAdamV[o] + (1.0f - beta2) * g * g;
                outputBias[o] += learningRate
                        * (outputBiasAdamM[o] / correction1)
                        / ((float) Math.sqrt(outputBiasAdamV[o] / correction2) + epsilon);
                outputBiasGradients[o] = 0.0f;
            }
        }

        private static float gaussian(Random random) {
            double u1 = Math.max(Double.MIN_VALUE, random.nextDouble());
            double u2 = random.nextDouble();
            return (float) (
                    Math.sqrt(-2.0 * Math.log(u1))
                            * Math.cos(2.0 * Math.PI * u2)
            );
        }

        private static float gaussianLogProbability(float value, float mean, float logStd) {
            float standard = (value - mean) / (float) Math.exp(logStd);
            return -0.5f * standard * standard
                    - logStd
                    - 0.5f * (float) Math.log(2.0 * Math.PI);
        }

        private static float squashedGaussianLogProbability(
                float action,
                float preTanh,
                float mean,
                float logStd
        ) {
            return gaussianLogProbability(preTanh, mean, logStd)
                    - (float) Math.log(Math.max(1e-6, 1.0 - action * action));
        }

        private static float atanh(float x) {
            return 0.5f * (float) Math.log((1.0f + x) / (1.0f - x));
        }

        public static final class Action {
            public final float yaw;
            public final float pitch;
            public final float logProbability;
            public final float value;

            Action(float yaw, float pitch, float logProbability, float value) {
                this.yaw = yaw;
                this.pitch = pitch;
                this.logProbability = logProbability;
                this.value = value;
            }
        }

        private record ForwardPass(
                float[] preActivation,
                float[] hidden,
                float[] output
        ) {}

        public record TrainingResult(
                float policyLoss,
                float valueLoss,
                float entropy
        ) {}
    }
}
