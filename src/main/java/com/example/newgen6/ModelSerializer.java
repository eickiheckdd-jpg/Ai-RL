package com.example.newgen6;

import java.io.*;

public class ModelSerializer {

    public static void saveModel(DoubleDQNAgent agent, String filepath) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filepath))) {
            dos.writeFloat(agent.getEpsilon());
            dos.writeInt(agent.getTrainingStepCount());

            float[][] weights = agent.qNetwork.getWeights();
            float[][] biases = agent.qNetwork.getBiases();

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
            if (numLayers != weights.length) {
                System.err.println("[Newgen6] ⚠️ Model architecture mismatch! Discarding old weights.");
                return false;
            }

            for (int i = 0; i < numLayers; i++) {
                int wLen = dis.readInt();
                if (wLen != weights[i].length) {
                    System.err.println("[Newgen6] ⚠️ Weight array size mismatch! Discarding old weights.");
                    return false;
                }
                for (int j = 0; j < wLen; j++) {
                    weights[i][j] = dis.readFloat();
                }
                
                int bLen = dis.readInt();
                if (bLen != biases[i].length) {
                    System.err.println("[Newgen6] ⚠️ Bias array size mismatch! Discarding old weights.");
                    return false;
                }
                for (int j = 0; j < bLen; j++) {
                    biases[i][j] = dis.readFloat();
                }
            }
            
            agent.targetNetwork.copyWeightsFrom(agent.qNetwork);
            System.out.println("[Newgen6] Model loaded successfully from " + filepath);
            return true;
        } catch (Exception e) {
            System.err.println("[Newgen6] Failed to load model (corrupt or outdated format): " + e.getMessage());
            return false;
        }
    }
}
