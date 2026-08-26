package com.example.newgen6.rl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class BrainStorage {
    private static final Path SAVE_PATH = Path.of("config", "newgen6_brain.json");

    public static void saveBrain(PolicyNetwork policy) {
        try {
            Files.createDirectories(SAVE_PATH.getParent());
            BufferedWriter writer = Files.newBufferedWriter(SAVE_PATH);
            
            writer.write("{\n");
            writer.write("  \"wTrunk\": " + matrixToJson(policy.getWTrunk()) + ",\n");
            writer.write("  \"wMove\": " + matrixToJson(policy.getWMove()) + ",\n");
            writer.write("  \"wToggle\": " + matrixToJson(policy.getWToggle()) + ",\n");
            writer.write("  \"wYaw\": " + matrixToJson(policy.getWYaw()) + ",\n");
            writer.write("  \"wPitch\": " + matrixToJson(policy.getWPitch()) + "\n");
            writer.write("}\n");
            writer.close();
            
            System.out.println("[NEWGEN6] Brain weights successfully saved to disk.");
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Failed to save brain weights: " + e.getMessage());
        }
    }

    private static String matrixToJson(float[][] matrix) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < matrix.length; i++) {
            sb.append("[");
            for (int j = 0; j < matrix[i].length; j++) {
                sb.append(matrix[i][j]);
                if (j < matrix[i].length - 1) sb.append(",");
            }
            sb.append("]");
            if (i < matrix.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }
}
