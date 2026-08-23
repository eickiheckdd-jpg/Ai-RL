package com.example.newgen6;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class PPOAgent {
    private static PPOAgent instance;
    private MultiLayerNetwork network;

    private PPOAgent() {
        // Enable workspace memory reuse explicitly for mobile
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(RLConfig.LEARNING_RATE))
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(RLConfig.OBS_DIM)
                        .nOut(16) // Minimal neuron count for 1.5GB RAM
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(16)
                        .nOut(RLConfig.ACTION_CONTINUOUS_DIM)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        network = new MultiLayerNetwork(conf);
        network.init();
    }

    public static synchronized PPOAgent getInstance() {
        if (instance == null) {
            instance = new PPOAgent();
        }
        return instance;
    }

    public float[] predict(float[] obs) {
        float[] actions = new float[RLConfig.ACTION_CONTINUOUS_DIM];
        
        // Wrap array creation inside an explicit ND4J Workspace to prevent memory leaks
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("MOBILE_AI_INFERENCE")) {
            INDArray input = Nd4j.create(obs).reshape(1, RLConfig.OBS_DIM);
            INDArray output = network.output(input, false);
            
            actions[0] = output.getFloat(0, 0) * RLConfig.MAX_YAW_DELTA;
            actions[1] = output.getFloat(0, 1) * RLConfig.MAX_PITCH_DELTA;
        }
        
        return actions;
    }
}
