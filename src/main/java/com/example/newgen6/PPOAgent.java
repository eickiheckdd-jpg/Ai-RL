package com.example.newgen6;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class PPOAgent {
    private static PPOAgent instance;
    private ComputationGraph network;

    // A Java 21 Record automatically generates the yawDelta() and pitchDelta() methods for you
    public record InferenceResult(float yawDelta, float pitchDelta) {}

    private PPOAgent() {
        // Rebuilt as a ComputationGraph to satisfy PPOTrainerThread
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(RLConfig.LEARNING_RATE))
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(RLConfig.OBS_DIM)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build(), "input")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(16)
                        .nOut(RLConfig.ACTION_CONTINUOUS_DIM)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build(), "dense1")
                .setOutputs("output")
                .build();

        network = new ComputationGraph(conf);
        network.init();
    }

    public static synchronized PPOAgent getInstance() {
        if (instance == null) {
            instance = new PPOAgent();
        }
        return instance;
    }

    public ComputationGraph getNetwork() {
        return network;
    }

    public InferenceResult predict(float[] obs) {
        float yaw = 0f;
        float pitch = 0f;
        
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("MOBILE_AI_INFERENCE")) {
            INDArray input = Nd4j.create(obs).reshape(1, RLConfig.OBS_DIM);
            INDArray[] output = network.output(false, input);
            
            yaw = output[0].getFloat(0, 0) * RLConfig.MAX_YAW_DELTA;
            pitch = output[0].getFloat(0, 1) * RLConfig.MAX_PITCH_DELTA;
        }
        
        return new InferenceResult(yaw, pitch);
    }
}
