package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public final class ModelRunner {

    private static final String MODEL_PATH =
            "/newgen6/newgen6_full.onnx";

    private static OrtEnvironment environment;
    private static OrtSession session;

    private ModelRunner() {
    }

    public static void initialize() {
        try {
            System.out.println("================================");
            System.out.println("NewGen6: Loading ONNX model...");
            System.out.println("================================");

            environment = OrtEnvironment.getEnvironment();

            byte[] modelBytes;

            try (InputStream input =
                         ModelRunner.class.getResourceAsStream(MODEL_PATH)) {

                if (input == null) {
                    throw new IOException(
                            "Model not found: " + MODEL_PATH
                    );
                }

                modelBytes = input.readAllBytes();
            }

            OrtSession.SessionOptions options =
                    new OrtSession.SessionOptions();

            session = environment.createSession(
                    modelBytes,
                    options
            );

            System.out.println();
            System.out.println("=== NewGen6 ONNX Model ===");

            System.out.println();
            System.out.println("INPUTS:");

            for (Map.Entry<String, NodeInfo> entry
                    : session.getInputInfo().entrySet()) {

                System.out.println(
                        "  " +
                        entry.getKey() +
                        " -> " +
                        entry.getValue().getInfo()
                );
            }

            System.out.println();
            System.out.println("OUTPUTS:");

            for (Map.Entry<String, NodeInfo> entry
                    : session.getOutputInfo().entrySet()) {

                System.out.println(
                        "  " +
                        entry.getKey() +
                        " -> " +
                        entry.getValue().getInfo()
                );
            }

            System.out.println();
            System.out.println("==========================");
            System.out.println("NewGen6: ONNX model loaded!");
            System.out.println("==========================");

        } catch (Exception e) {

            System.err.println();
            System.err.println(
                    "NewGen6: FAILED TO LOAD ONNX MODEL"
            );
            System.err.println();

            e.printStackTrace();
        }
    }

    public static boolean isLoaded() {
        return session != null;
    }

    public static OrtSession getSession() {
        return session;
    }

    public static OrtEnvironment getEnvironment() {
        return environment;
    }

    public static void close() {

        try {
            if (session != null) {
                session.close();
                session = null;
            }
        } finally {
            if (environment != null) {
                environment.close();
                environment = null;
            }
        }
    }
}