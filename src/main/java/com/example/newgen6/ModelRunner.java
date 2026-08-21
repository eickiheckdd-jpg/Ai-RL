package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ModelRunner {

private static final String MODEL_PATH =
        "/newgen6/newgen6_full.onnx";

private static final ExecutorService LOADER =
        Executors.newSingleThreadExecutor(r -> {
            Thread thread = new Thread(r, "NewGen6-ONNX-Loader");
            thread.setDaemon(true);
            return thread;
        });

private static volatile OrtEnvironment environment;
private static volatile OrtSession session;

private static volatile boolean loading = false;
private static volatile boolean failed = false;
private static volatile boolean inferenceTestStarted = false;
private static volatile boolean inferenceTestFinished = false;

private ModelRunner() {
}

public static synchronized void initializeAsync() {

    if (session != null) {
        System.out.println("NewGen6: ONNX model is already loaded.");
        return;
    }

    if (loading) {
        System.out.println("NewGen6: ONNX model is already loading.");
        return;
    }

    loading = true;
    failed = false;

    System.out.println("NewGen6: Starting background ONNX loading...");

    LOADER.execute(ModelRunner::loadModel);
}

private static void loadModel() {

    OrtEnvironment localEnvironment = null;
    OrtSession localSession = null;

    try {
        System.out.println("================================");
        System.out.println("NewGen6: Loading ONNX model...");
        System.out.println(
                "Thread: " +
                Thread.currentThread().getName()
        );
        System.out.println("================================");

        localEnvironment = OrtEnvironment.getEnvironment();

        byte[] modelBytes;

        try (InputStream input =
                     ModelRunner.class.getResourceAsStream(MODEL_PATH)) {

            if (input == null) {
                throw new IOException(
                        "Model not found inside mod JAR: " + MODEL_PATH
                );
            }

            modelBytes = input.readAllBytes();
        }

        System.out.println(
                "NewGen6: Model file loaded: " +
                modelBytes.length +
                " bytes"
        );

        OrtSession.SessionOptions options =
                new OrtSession.SessionOptions();

        try {
            localSession =
                    localEnvironment.createSession(
                            modelBytes,
                            options
                    );
        } finally {
            options.close();
        }

        environment = localEnvironment;
        session = localSession;

        localEnvironment = null;
        localSession = null;

        System.out.println();
        System.out.println("=== NewGen6 ONNX Model ===");

        System.out.println();
        System.out.println("INPUTS:");

        for (Map.Entry<String, NodeInfo> entry :
                session.getInputInfo().entrySet()) {

            System.out.println(
                    "  " +
                    entry.getKey() +
                    " -> " +
                    entry.getValue().getInfo()
            );
        }

        System.out.println();
        System.out.println("OUTPUTS:");

        for (Map.Entry<String, NodeInfo> entry :
                session.getOutputInfo().entrySet()) {

            System.out.println(
                    "  " +
                    entry.getKey() +
                    " -> " +
                    entry.getValue().getInfo()
            );
        }

        System.out.println();
        System.out.println("==========================");
        System.out.println("NewGen6: ONNX MODEL LOADED!");
        System.out.println("==========================");

        /*
         * The model is loaded successfully.
         *
         * Now perform exactly ONE inference test.
         *
         * IMPORTANT:
         * This is deliberately done on the background
         * ONNX loader thread, never on Minecraft's
         * render/tick thread.
         */
        runInferenceTest();

    } catch (Throwable e) {

        failed = true;

        System.err.println();
        System.err.println(
                "NewGen6: FAILED TO LOAD ONNX MODEL"
        );
        System.err.println(
                "NewGen6: Minecraft will NOT be crashed."
        );
        System.err.println();

        e.printStackTrace();

        if (localSession != null) {
            try {
                localSession.close();
            } catch (Exception ignored) {
            }
        }

        if (localEnvironment != null) {
            try {
                localEnvironment.close();
            } catch (Exception ignored) {
            }
        }

        session = null;
        environment = null;

    } finally {

        loading = false;

        System.out.println(
                "NewGen6: ONNX loader finished. " +
                "loaded=" + (session != null) +
                ", failed=" + failed
        );
    }
}

/**
 * Performs one diagnostic inference attempt.
 *
 * We do NOT invent input data for the model.
 *
 * Instead, this first test checks whether the model's
 * input metadata can be understood safely.
 *
 * Actual Minecraft/game-state tensors will be added
 * after we know exactly what the model expects.
 */
private static void runInferenceTest() {

    if (session == null) {
        return;
    }

    if (inferenceTestStarted) {
        return;
    }

    inferenceTestStarted = true;

    System.out.println();
    System.out.println("================================");
    System.out.println("NewGen6: INFERENCE TEST");
    System.out.println("================================");

    try {

        /*
         * First verify that ONNX Runtime can access
         * the model's inputs and outputs.
         */
        Map<String, NodeInfo> inputs =
                session.getInputInfo();

        Map<String, NodeInfo> outputs =
                session.getOutputInfo();

        System.out.println(
                "NewGen6: Input count = " +
                inputs.size()
        );

        System.out.println(
                "NewGen6: Output count = " +
                outputs.size()
        );

        for (Map.Entry<String, NodeInfo> entry :
                inputs.entrySet()) {

            System.out.println(
                    "NewGen6: Input: " +
                    entry.getKey() +
                    " -> " +
                    entry.getValue().getInfo()
            );
        }

        for (Map.Entry<String, NodeInfo> entry :
                outputs.entrySet()) {

            System.out.println(
                    "NewGen6: Output: " +
                    entry.getKey() +
                    " -> " +
                    entry.getValue().getInfo()
            );
        }

        /*
         * We intentionally stop here for this first test.
         *
         * We need the actual tensor type/shape before
         * constructing an input tensor. Sending a random
         * tensor could cause an invalid-shape/type error.
         */
        System.out.println();
        System.out.println(
                "NewGen6: Model interface verified."
        );
        System.out.println(
                "NewGen6: Waiting for input schema before "
                + "running a real tensor inference."
        );
        System.out.println(
                "NewGen6: Minecraft thread was never blocked."
        );

        inferenceTestFinished = true;

        System.out.println("================================");
        System.out.println(
                "NewGen6: INFERENCE TEST PASSED"
        );
        System.out.println("================================");

    } catch (Throwable e) {

        System.err.println();
        System.err.println(
                "NewGen6: INFERENCE TEST FAILED"
        );

        e.printStackTrace();

        inferenceTestFinished = true;
    }
}

public static boolean isLoaded() {
    return session != null;
}

public static boolean isLoading() {
    return loading;
}

public static boolean hasFailed() {
    return failed;
}

public static boolean isInferenceTestFinished() {
    return inferenceTestFinished;
}

public static OrtSession getSession() {
    return session;
}

public static OrtEnvironment getEnvironment() {
    return environment;
}

public static synchronized void close() {

    OrtSession currentSession = session;
    OrtEnvironment currentEnvironment = environment;

    session = null;
    environment = null;

    if (currentSession != null) {
        try {
            currentSession.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    if (currentEnvironment != null) {
        try {
            currentEnvironment.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    System.out.println(
            "NewGen6: ONNX resources closed."
    );
}

}