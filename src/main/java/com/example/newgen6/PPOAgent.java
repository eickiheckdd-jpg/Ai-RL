package com.example.newgen6;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class PPOAgent {
    private final ComputationGraph network;

    // Load existing or init fresh
    public PPOAgent(File modelFile) {
        if (modelFile.exists()) {
            System.out.println("[NewGen6] Found existing PPO Brain. Loading from " + modelFile.getAbsolutePath());
            ComputationGraph loadedNet = null;
            try {
                loadedNet = ModelSerializer.restoreComputationGraph(modelFile);
            } catch (Exception e) {
                System.err.println("[NewGen6] Failed to load brain, generating a new one...");
                e.printStackTrace();
            }
            this.network = loadedNet != null ? loadedNet : buildNewNetwork();
        } else {
            System.out.println("[NewGen6] No existing brain found. Initializing fresh PPO Agent.");
            this.network = buildNewNetwork();
        }
    }

    private ComputationGraph buildNewNetwork() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(1337)
            .updater(new Adam(RLConfig.LEARNING_RATE))
            .weightInit(WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("input")
            .addLayer("dense1", new DenseLayer.Builder().nIn(RLConfig.OBS_DIM).nOut(128).activation(Activation.RELU).build(), "input")
            .addLayer("dense2", new DenseLayer.Builder().nIn(128).nOut(128).activation(Activation.RELU).build(), "dense1")
            
            // Aim-only Policy Head (Dim: 2)
            .addLayer("policy_head", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(128).nOut(RLConfig.ACTION_CONTINUOUS_DIM)
                .activation(Activation.TANH).build(), "dense2")
                
            // Value Head (Scalar)
            .addLayer("value_head", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(128).nOut(1)
                .activation(Activation.IDENTITY).build(), "dense2")
                
            .setOutputs("policy_head", "value_head")
            .setInputTypes(InputType.feedForward(RLConfig.OBS_DIM))
            .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;
    }

    public InferenceResult step(float[] obs) {
        INDArray inputTensor = Nd4j.create(obs).reshape(1, RLConfig.OBS_DIM);
        INDArray[] outputs = network.output(false, inputTensor);
        
        INDArray policyOut = outputs[0];
        INDArray valueOut = outputs[1];

        float yawDelta = clamp(policyOut.getFloat(0, 0) * RLConfig.MAX_YAW_DELTA);
        float pitchDelta = clamp(policyOut.getFloat(0, 1) * RLConfig.MAX_PITCH_DELTA);
        
        float valEstimate = Float.isNaN(valueOut.getFloat(0, 0)) ? 0.0f : valueOut.getFloat(0, 0);

        return new InferenceResult(yawDelta, pitchDelta, valEstimate);
    }

    public void saveBrain(File modelFile) {
        System.out.println("[NewGen6] Saving PPO Brain to " + modelFile.getAbsolutePath());
        try {
            ModelSerializer.writeModel(network, modelFile, true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public ComputationGraph getNetwork() { return network; }

    private float clamp(float val) {
        return (Float.isNaN(val) || Float.isInfinite(val)) ? 0.0f : val;
    }

    public record InferenceResult(float yawDelta, float pitchDelta, float valueEstimate) {}
}
