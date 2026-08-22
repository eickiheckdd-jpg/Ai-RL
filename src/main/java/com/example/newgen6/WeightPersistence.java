package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Binary weight persistence — plain DataOutputStream/DataInputStream, no
 * Gson/JSON library, no external dependency. Saves both actor and critic
 * into a single file under .minecraft/config/newgen6/.
 */
public class WeightPersistence {

    public static void save(Path configDir, PPOBrain brain) {
        try {
            Path dir = configDir.resolve("newgen6");
            Files.createDirectories(dir);
            Path file = dir.resolve("ppo_weights.bin");

            try (DataOutputStream out = new DataOutputStream(
                    new BufferedOutputStream(new FileOutputStream(file.toFile())))) {
                writeNetwork(out, brain.getActor());
                writeNetwork(out, brain.getCritic());
            }
        } catch (IOException e) {
            // Non-fatal — training continues in-memory even if the save fails.
            e.printStackTrace();
        }
    }

    public static boolean load(Path configDir, PPOBrain brain) {
        Path file = configDir.resolve("newgen6").resolve("ppo_weights.bin");
        if (!Files.exists(file)) return false;

        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(file.toFile())))) {
            readNetwork(in, brain.getActor());
            readNetwork(in, brain.getCritic());
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void writeNetwork(DataOutputStream out, MLP net) throws IOException {
        double[][][] weights = net.getWeights();
        double[][] biases = net.getBiases();

        out.writeInt(weights.length); // number of layers
        for (int l = 0; l < weights.length; l++) {
            out.writeInt(weights[l].length);      // input size for this layer
            out.writeInt(weights[l][0].length);   // output size for this layer
            for (double[] row : weights[l]) {
                for (double v : row) out.writeDouble(v);
            }
            for (double b : biases[l]) out.writeDouble(b);
        }
    }

    private static void readNetwork(DataInputStream in, MLP net) throws IOException {
        double[][][] weights = net.getWeights();
        double[][] biases = net.getBiases();

        int numLayers = in.readInt();
        if (numLayers != weights.length) {
            throw new IOException("Saved network layer count (" + numLayers
                    + ") does not match current architecture (" + weights.length
                    + "). Did BotState.FEATURE_COUNT or BotAction.COUNT change?");
        }

        for (int l = 0; l < numLayers; l++) {
            int inSize = in.readInt();
            int outSize = in.readInt();
            if (inSize != weights[l].length || outSize != weights[l][0].length) {
                throw new IOException("Saved layer shape mismatch at layer " + l);
            }
            for (int i = 0; i < inSize; i++) {
                for (int j = 0; j < outSize; j++) {
                    weights[l][i][j] = in.readDouble();
                }
            }
            for (int j = 0; j < outSize; j++) {
                biases[l][j] = in.readDouble();
            }
        }
    }
}
