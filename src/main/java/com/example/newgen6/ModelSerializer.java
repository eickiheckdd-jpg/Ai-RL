package com.example.newgen6;

import java.io.*;

public class ModelSerializer {
    private static final int FILE_VERSION = 1;

    public static void saveAgent(DoubleDQNAgent agent, File file) {
        File tempFile = new File(file.getAbsolutePath() + ".tmp");
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tempFile)))) {
            dos.writeInt(FILE_VERSION);
            dos.writeFloat(agent.getEpsilon());
            dos.writeInt(agent.getStepCount());

            float[][] weights = agent.getOnlineNetwork().getWeights();
            float[][] biases = agent.getOnlineNetwork().getBiases();

            dos.writeInt(weights.length);
            for (int i = 0; i < weights.length; i++) {
                dos.writeInt(weights[i].length);
                for (float w : weights[i]) dos.writeFloat(w);

                dos.writeInt(biases[i].length);
                for (float b : biases[i]) dos.writeFloat(b);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        if (file.exists()) file.delete();
        tempFile.renameTo(file);
    }

    public static boolean loadAgent(DoubleDQNAgent agent, File file) {
        if (!file.exists()) return false;

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            int version = dis.readInt();
            if (version != FILE_VERSION) return false;

            agent.setEpsilon(dis.readFloat());
            agent.setStepCount(dis.readInt());

            int numLayers = dis.readInt();
            float[][] weights = agent.getOnlineNetwork().getWeights();
            float[][] biases = agent.getOnlineNetwork().getBiases();

            if (numLayers != weights.length) return false;

            for (int i = 0; i < numLayers; i++) {
                int wLen = dis.readInt();
                if (wLen != weights[i].length) return false;
                for (int j = 0; j < wLen; j++) weights[i][j] = dis.readFloat();

                int bLen = dis.readInt();
                if (bLen != biases[i].length) return false;
                for (int j = 0; j < bLen; j++) biases[i][j] = dis.readFloat();
            }

            // Sync target network correctly from online network
            agent.getTargetNetwork().copyWeightsFrom(agent.getOnlineNetwork());

            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
