package com.example.newgen6;

import java.io.*;

public class ModelSerializer {

    public static void saveModel(DoubleDQNAgent agent, String filepath) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filepath))) {
            // Save the epsilon and training step count
            dos.writeFloat(agent.getEpsilon());
            dos.writeInt(agent.getTrainingStepCount());

            // Save weights and biases from the main qNetwork
            float[][] weights = agent.qNetwork.getWeights();
            float[][] biases = agent.qNetwork.getBiases();

            // Write lengths for dynamic reconstruction if needed
            dos.writeInt(weights.length);
            for (int i = 0; i < weights.length; i++) {
                dos.writeInt(weights[i].length);
                for (int j = 0; j < weights[i].length; j++) {
                    dos.writeFloat(weights[i][j]);
                }
                
                dos.writeInt(biases[i].length);
                for (int j = 0; j < biases[i].length; j++) {
                    dos.writeFloat(biases[i][j]);
                }
            }
            System.out.println("[Newgen6] Model saved successfully to " + filepath);
        } catch (IOException e) {
            System.err.println("[Newgen6] Failed to save model: " + e.getMessage());
        }
    }

    public static boolean loadModel(DoubleDQNAgent agent, String filepath) {
        File file = new File(filepath);
        if (!file.exists()) return false;

        try (DataInputStream dis = new DataInputStream(new FileInputStream(filepath))) {
            agent.setEpsilon(dis.readFloat());
            agent.setTrainingStepCount(dis.readInt());

            float[][] weights = agent.qNetwork.getWeights();
            float[][] biases = agent.qNetwork.getBiases();

            int numLayers = dis.readInt();
            for (int i = 0; i < numLayers; i++) {
                int wLen = dis.readInt();
                for (int j = 0; j < wLen; j++) {
                    weights[i][j] = dis.readFloat();
                }
                
                int bLen = dis.readInt();
                for (int j = 0; j < bLen; j++) {
                    biases[i][j] = dis.readFloat();
                }
            }
            
            // Sync the target network with the loaded weights
            agent.targetNetwork.copyWeightsFrom(agent.qNetwork);
            
            System.out.println("[Newgen6] Model loaded successfully from " + filepath);
            return true;
        } catch (IOException e) {
            System.err.println("[Newgen6] Failed to load model: " + e.getMessage());
            return false;
        }
    }
}
