package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
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

    private ModelRunner() {
    }

    /**
     * Starts model loading in a background thread.
     *
     * This method returns immediately and NEVER blocks
     * the Minecraft client thread.
     */
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

    /**
     * Actually loads the ONNX model.
     *
     * IMPORTANT:
     * This runs on the NewGen6-ONNX-Loader thread,
     * NOT the Minecraft client thread.
     */
    private static void loadModel() {

        OrtEnvironment localEnvironment = null;
        OrtSession localSession = null;

        try {
            System.out.println("================================");
            System.out.println("NewGen6: Loading ONNX model...");
            System.out.println("Thread: " + Thread.currentThread().getName());
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
                localSession = localEnvironment.createSession(
                        modelBytes,
                        options
                );
            } finally {
                options.close();
            }

            /*
             * Only publish the session after it has been
             * completely created successfully.
             */
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

    public static boolean isLoaded() {
        return session != null;
    }

    public static boolean isLoading() {
        return loading;
    }

    public static boolean hasFailed() {
        return failed;
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

        System.out.println("NewGen6: ONNX resources closed.");
    }
}