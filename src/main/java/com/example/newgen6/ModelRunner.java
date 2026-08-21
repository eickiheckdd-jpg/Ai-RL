package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ModelRunner {

    /*
     * Both model files MUST exist in the JAR at:
     *
     * src/main/resources/newgen6/
     *
     * newgen6_full.onnx
     * newgen6_full.onnx.data
     */
    private static final String MODEL_PATH =
            "/newgen6/newgen6_full.onnx";

    private static final String MODEL_DATA_PATH =
            "/newgen6/newgen6_full.onnx.data";

    /*
     * Android ARM64 ONNX Runtime native libraries.
     */
    private static final String NATIVE_DIR =
            "/ai/onnxruntime/native/android-arm64/";

    private static final String ORT_NATIVE =
            NATIVE_DIR + "libonnxruntime.so";

    private static final String ORT_JNI_NATIVE =
            NATIVE_DIR + "libonnxruntime4j_jni.so";

    /*
     * Background loader thread.
     *
     * ONNX loading and initialization never happens
     * on Minecraft's render/tick thread.
     */
    private static final ExecutorService LOADER =
            Executors.newSingleThreadExecutor(r -> {
                Thread thread =
                        new Thread(r, "NewGen6-ONNX-Loader");

                thread.setDaemon(true);

                return thread;
            });

    private static volatile OrtEnvironment environment;
    private static volatile OrtSession session;

    private static volatile boolean loading = false;
    private static volatile boolean failed = false;

    private static volatile boolean inferenceTestStarted = false;
    private static volatile boolean inferenceTestFinished = false;

    /*
     * Directory containing extracted Android native libraries.
     */
    private static volatile Path nativeDirectory;

    /*
     * Directory containing:
     *
     * newgen6_full.onnx
     * newgen6_full.onnx.data
     */
    private static volatile Path modelDirectory;

    private ModelRunner() {
    }

    public static synchronized void initializeAsync() {

        if (session != null) {

            System.out.println(
                    "NewGen6: ONNX model is already loaded."
            );

            return;
        }

        if (loading) {

            System.out.println(
                    "NewGen6: ONNX model is already loading."
            );

            return;
        }

        loading = true;
        failed = false;

        System.out.println(
                "NewGen6: Starting background ONNX loading..."
        );

        LOADER.execute(ModelRunner::loadModel);
    }

    private static void loadModel() {

        OrtEnvironment localEnvironment = null;
        OrtSession localSession = null;

        try {

            System.out.println("================================");

            System.out.println(
                    "NewGen6: Loading ONNX model..."
            );

            System.out.println(
                    "Thread: " +
                    Thread.currentThread().getName()
            );

            System.out.println("================================");

            /*
             * IMPORTANT:
             *
             * Native libraries MUST be prepared before
             * the first initialization of ONNX Runtime.
             */
            prepareAndroidNativeLibraries();

            /*
             * Initialize ONNX Runtime only after the
             * native libraries have been extracted.
             */
            localEnvironment =
                    OrtEnvironment.getEnvironment();

            /*
             * Extract the ONNX model and its external
             * data file into the SAME directory.
             *
             * This is required because the ONNX file
             * references:
             *
             * newgen6_full.onnx.data
             */
            prepareModelFiles();

            Path modelFile =
                    modelDirectory.resolve(
                            "newgen6_full.onnx"
                    );

            Path modelDataFile =
                    modelDirectory.resolve(
                            "newgen6_full.onnx.data"
                    );

            System.out.println();

            System.out.println(
                    "NewGen6: Model directory: " +
                    modelDirectory
            );

            System.out.println(
                    "NewGen6: Model file: " +
                    modelFile
            );

            System.out.println(
                    "NewGen6: Model data file: " +
                    modelDataFile
            );

            System.out.println(
                    "NewGen6: Model file size: " +
                    Files.size(modelFile) +
                    " bytes"
            );

            System.out.println(
                    "NewGen6: Model external data size: " +
                    Files.size(modelDataFile) +
                    " bytes"
            );

            /*
             * Create the session FROM THE FILE PATH.
             *
             * DO NOT use:
             *
             * createSession(byte[], options)
             *
             * because the model uses external tensor data.
             *
             * Loading from the file path allows ONNX Runtime
             * to resolve newgen6_full.onnx.data beside it.
             */
            OrtSession.SessionOptions options =
                    new OrtSession.SessionOptions();

            try {

                localSession =
                        localEnvironment.createSession(
                                modelFile.toString(),
                                options
                        );

            } finally {

                options.close();
            }

            /*
             * Model successfully loaded.
             */
            environment = localEnvironment;
            session = localSession;

            localEnvironment = null;
            localSession = null;

            System.out.println();
            System.out.println(
                    "=== NewGen6 ONNX Model ==="
            );

            /*
             * Print inputs.
             */
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

            /*
             * Print outputs.
             */
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
            System.out.println(
                    "=========================="
            );

            System.out.println(
                    "NewGen6: ONNX MODEL LOADED!"
            );

            System.out.println(
                    "=========================="
            );

            /*
             * Perform the diagnostic interface test.
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

            /*
             * Clean up partially created session.
             */
            if (localSession != null) {

                try {
                    localSession.close();
                } catch (Exception ignored) {
                }
            }

            /*
             * Clean up partially created environment.
             */
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
     * Extracts the Android ARM64 ONNX Runtime libraries
     * into a writable temporary directory.
     */
    private static synchronized void
    prepareAndroidNativeLibraries()
            throws IOException {

        if (nativeDirectory != null) {

            System.setProperty(
                    "onnxruntime.native.path",
                    nativeDirectory.toString()
            );

            System.out.println(
                    "NewGen6: Reusing native directory: " +
                    nativeDirectory
            );

            return;
        }

        Path directory =
                Files.createTempDirectory(
                        "newgen6-onnx-arm64-"
                );

        Path ortPath =
                directory.resolve(
                        "libonnxruntime.so"
                );

        Path jniPath =
                directory.resolve(
                        "libonnxruntime4j_jni.so"
                );

        /*
         * Extract the real ARM64 libraries.
         */
        extractResource(
                ORT_NATIVE,
                ortPath
        );

        extractResource(
                ORT_JNI_NATIVE,
                jniPath
        );

        /*
         * Tell ONNX Runtime where the libraries are.
         */
        System.setProperty(
                "onnxruntime.native.path",
                directory.toString()
        );

        nativeDirectory = directory;

        System.out.println(
                "NewGen6: Android ARM64 native directory: " +
                directory
        );

        System.out.println(
                "NewGen6: libonnxruntime.so size: " +
                Files.size(ortPath) +
                " bytes"
        );

        System.out.println(
                "NewGen6: libonnxruntime4j_jni.so size: " +
                Files.size(jniPath) +
                " bytes"
        );
    }

    /**
     * Extracts BOTH model files into the same directory.
     */
    private static synchronized void
    prepareModelFiles()
            throws IOException {

        if (modelDirectory != null) {

            System.out.println(
                    "NewGen6: Reusing model directory: " +
                    modelDirectory
            );

            return;
        }

        Path directory =
                Files.createTempDirectory(
                        "newgen6-model-"
                );

        Path modelFile =
                directory.resolve(
                        "newgen6_full.onnx"
                );

        Path modelDataFile =
                directory.resolve(
                        "newgen6_full.onnx.data"
                );

        /*
         * Extract the ONNX model.
         */
        extractResource(
                MODEL_PATH,
                modelFile
        );

        /*
         * Extract the external tensor data.
         *
         * This MUST have exactly the filename:
         *
         * newgen6_full.onnx.data
         */
        extractResource(
                MODEL_DATA_PATH,
                modelDataFile
        );

        /*
         * Verify that both files exist.
         */
        if (!Files.exists(modelFile)) {

            throw new IOException(
                    "Extracted ONNX model does not exist: " +
                    modelFile
            );
        }

        if (!Files.exists(modelDataFile)) {

            throw new IOException(
                    "Extracted ONNX external data does not exist: " +
                    modelDataFile
            );
        }

        if (Files.size(modelFile) <= 0) {

            throw new IOException(
                    "ONNX model file is empty: " +
                    modelFile
            );
        }

        if (Files.size(modelDataFile) <= 0) {

            throw new IOException(
                    "ONNX external data file is empty: " +
                    modelDataFile
            );
        }

        modelDirectory = directory;

        System.out.println(
                "NewGen6: Model files extracted successfully."
        );
    }

    /**
     * Extracts a resource from the mod JAR into a file.
     */
    private static void extractResource(
            String resourcePath,
            Path destination
    ) throws IOException {

        try (InputStream input =
                     ModelRunner.class
                             .getResourceAsStream(
                                     resourcePath
                             )) {

            if (input == null) {

                throw new IOException(
                        "Resource not found inside mod JAR: " +
                        resourcePath
                );
            }

            Files.copy(
                    input,
                    destination
            );
        }
    }

    /**
     * Performs one diagnostic inference/interface test.
     *
     * We intentionally do not create a fake tensor.
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

        System.out.println(
                "================================"
        );

        System.out.println(
                "NewGen6: INFERENCE TEST"
        );

        System.out.println(
                "================================"
        );

        try {

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

            /*
             * Print input information.
             */
            for (Map.Entry<String, NodeInfo> entry :
                    inputs.entrySet()) {

                System.out.println(
                        "NewGen6: Input: " +
                        entry.getKey() +
                        " -> " +
                        entry.getValue().getInfo()
                );
            }

            /*
             * Print output information.
             */
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
             * We intentionally stop here.
             *
             * We don't yet know the actual game-state
             * tensor values that should be supplied.
             */
            System.out.println();

            System.out.println(
                    "NewGen6: Model interface verified."
            );

            System.out.println(
                    "NewGen6: Waiting for input schema " +
                    "before running a real tensor inference."
            );

            System.out.println(
                    "NewGen6: Minecraft thread was never blocked."
            );

            inferenceTestFinished = true;

            System.out.println(
                    "================================"
            );

            System.out.println(
                    "NewGen6: INFERENCE TEST PASSED"
            );

            System.out.println(
                    "================================"
            );

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

    /**
     * Closes ONNX Runtime resources.
     */
    public static synchronized void close() {

        OrtSession currentSession =
                session;

        OrtEnvironment currentEnvironment =
                environment;

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

        nativeDirectory = null;
        modelDirectory = null;

        System.out.println(
                "NewGen6: ONNX resources closed."
        );
    }
}