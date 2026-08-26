package com.example.newgen6.rl;

import com.example.newgen6.rl.nn.DenseLayer;
import com.example.newgen6.rl.nn.GRUCell;
import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Random;

/**
 * Java-only binary checkpoint for the RL model parameters.
 *
 * No PT, ONNX, or JSON is used.
 *
 * The checkpoint stores network parameters only. Optimizer moments and
 * rollout buffers intentionally remain runtime state so a fresh optimizer
 * can be constructed safely after loading.
 */
public final class ModelCheckpoint {

    private static final int MAGIC = 0x4E473652; // "NG6R"
    private static final int FORMAT_VERSION = 1;

    private ModelCheckpoint() {
    }

    public static void save(
            Path file,
            PolicyValueNetwork network) throws IOException {

        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(network, "network");

        TrainingValidator.validateNetwork(network);

        Path parent = file.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        Path temporary = file.resolveSibling(
                file.getFileName() + ".tmp"
        );

        try (
                DataOutputStream out = new DataOutputStream(
                        new BufferedOutputStream(
                                Files.newOutputStream(temporary)
                        )
                )
        ) {
            out.writeInt(MAGIC);
            out.writeInt(FORMAT_VERSION);

            out.writeInt(RLConstants.OBSERVATION_SIZE);
            out.writeInt(RLConstants.GRU_HIDDEN_SIZE);
            out.writeInt(RLConstants.TRUNK_SIZE);

            writeDense(out, network.trunk());
            writeDense(out, network.moveHead());
            writeDense(out, network.yawHead());
            writeDense(out, network.pitchHead());
            writeDense(out, network.jumpHead());
            writeDense(out, network.sprintHead());
            writeDense(out, network.sneakHead());
            writeDense(out, network.attackHead());
            writeDense(out, network.valueHead());

            writeGru(out, network.gru());

            out.flush();
        }

        Files.move(
                temporary,
                file,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE
        );
    }

    public static void load(
            Path file,
            PolicyValueNetwork network) throws IOException {

        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(network, "network");

        try (
                DataInputStream in = new DataInputStream(
                        new BufferedInputStream(
                                Files.newInputStream(file)
                        )
                )
        ) {
            int magic = in.readInt();

            if (magic != MAGIC) {
                throw new IOException(
                        "Invalid NewGen6 checkpoint magic"
                );
            }

            int version = in.readInt();

            if (version != FORMAT_VERSION) {
                throw new IOException(
                        "Unsupported checkpoint format: "
                                + version
                );
            }

            int observations = in.readInt();
            int hidden = in.readInt();
            int trunk = in.readInt();

            if (observations != RLConstants.OBSERVATION_SIZE
                    || hidden != RLConstants.GRU_HIDDEN_SIZE
                    || trunk != RLConstants.TRUNK_SIZE) {

                throw new IOException(
                        "Checkpoint architecture does not match "
                                + "the current RLConstants ABI"
                );
            }

            readDense(in, network.trunk());
            readDense(in, network.moveHead());
            readDense(in, network.yawHead());
            readDense(in, network.pitchHead());
            readDense(in, network.jumpHead());
            readDense(in, network.sprintHead());
            readDense(in, network.sneakHead());
            readDense(in, network.attackHead());
            readDense(in, network.valueHead());

            readGru(in, network.gru());
        } catch (EOFException e) {
            throw new IOException(
                    "Checkpoint is truncated",
                    e
            );
        }

        TrainingValidator.validateNetwork(network);
    }

    private static void writeDense(
            DataOutputStream out,
            DenseLayer layer) throws IOException {

        out.writeInt(layer.inputSize());
        out.writeInt(layer.outputSize());
        out.writeInt(layer.activation().ordinal());

        float[][] weights = layer.weights();
        float[] bias = layer.bias();

        for (float[] row : weights) {
            for (float value : row) {
                out.writeFloat(value);
            }
        }

        for (float value : bias) {
            out.writeFloat(value);
        }
    }

    private static void readDense(
            DataInputStream in,
            DenseLayer layer) throws IOException {

        int inputSize = in.readInt();
        int outputSize = in.readInt();
        int activation = in.readInt();

        if (inputSize != layer.inputSize()
                || outputSize != layer.outputSize()
                || activation != layer.activation().ordinal()) {

            throw new IOException(
                    "Dense layer architecture mismatch"
            );
        }

        float[][] weights = layer.weights();
        float[] bias = layer.bias();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = in.readFloat();
            }
        }

        for (int i = 0; i < bias.length; i++) {
            bias[i] = in.readFloat();
        }
    }

    private static void writeGru(
            DataOutputStream out,
            GRUCell gru) throws IOException {

        out.writeInt(gru.getInputSize());
        out.writeInt(gru.getHiddenSize());

        writeMatrix(out, gru.weightsZ());
        writeMatrix(out, gru.recurrentWeightsZ());
        writeVector(out, gru.biasZ());

        writeMatrix(out, gru.weightsR());
        writeMatrix(out, gru.recurrentWeightsR());
        writeVector(out, gru.biasR());

        writeMatrix(out, gru.weightsN());
        writeMatrix(out, gru.recurrentWeightsN());
        writeVector(out, gru.biasN());
    }

    private static void readGru(
            DataInputStream in,
            GRUCell gru) throws IOException {

        int inputSize = in.readInt();
        int hiddenSize = in.readInt();

        if (inputSize != gru.getInputSize()
                || hiddenSize != gru.getHiddenSize()) {

            throw new IOException(
                    "GRU architecture mismatch"
            );
        }

        readMatrix(in, gru.weightsZ());
        readMatrix(in, gru.recurrentWeightsZ());
        readVector(in, gru.biasZ());

        readMatrix(in, gru.weightsR());
        readMatrix(in, gru.recurrentWeightsR());
        readVector(in, gru.biasR());

        readMatrix(in, gru.weightsN());
        readMatrix(in, gru.recurrentWeightsN());
        readVector(in, gru.biasN());
    }

    private static void writeMatrix(
            DataOutputStream out,
            float[][] matrix) throws IOException {

        out.writeInt(matrix.length);

        for (float[] row : matrix) {
            out.writeInt(row.length);

            for (float value : row) {
                out.writeFloat(value);
            }
        }
    }

    private static void readMatrix(
            DataInputStream in,
            float[][] matrix) throws IOException {

        int rows = in.readInt();

        if (rows != matrix.length) {
            throw new IOException(
                    "Matrix row-count mismatch"
            );
        }

        for (int i = 0; i < matrix.length; i++) {
            int columns = in.readInt();

            if (columns != matrix[i].length) {
                throw new IOException(
                        "Matrix column-count mismatch"
                );
            }

            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = in.readFloat();
            }
        }
    }

    private static void writeVector(
            DataOutputStream out,
            float[] vector) throws IOException {

        out.writeInt(vector.length);

        for (float value : vector) {
            out.writeFloat(value);
        }
    }

    private static void readVector(
            DataInputStream in,
            float[] vector) throws IOException {

        int length = in.readInt();

        if (length != vector.length) {
            throw new IOException(
                    "Vector length mismatch"
            );
        }

        for (int i = 0; i < vector.length; i++) {
            vector[i] = in.readFloat();
        }
    }
}