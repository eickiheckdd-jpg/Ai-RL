package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ModelRunner {

    private static final String MODEL_PATH =
            "/newgen6/newgen6_full.onnx";

    private static final String MODEL_DATA_PATH =
            "/newgen6/newgen6_full.onnx.data";

    private static final String NATIVE_DIR =
            "/ai/onnxruntime/native/android-arm64/";

    private static final String ORT_NATIVE =
            NATIVE_DIR + "libonnxruntime.so";

    private static final String ORT_JNI_NATIVE =
            NATIVE_DIR + "libonnxruntime4j_jni.so";

    private static final int FEATURE_COUNT = 78;
    private static final int MODEL_FEATURE_COUNT = 156;
    private static final int HISTORY_LENGTH = 96;

    private static final ExecutorService LOADER =
            Executors.newSingleThreadExecutor(r -> {
                Thread thread =
                        new Thread(r, "NewGen6-ONNX-Loader");
                thread.setDaemon(true);
                return thread;
            });

    private static volatile OrtEnvironment environment;
    private static volatile OrtSession session;

    private static volatile Path nativeDirectory;
    private static volatile Path modelDirectory;

    private static volatile boolean loading;
    private static volatile boolean failed;

    private ModelRunner() {
    }

    public static synchronized void initializeAsync() {

        if (session != null) {
            System.out.println(
                    "NewGen6: ONNX model already loaded."
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
             * Native libraries must be prepared before
             * ONNX Runtime is initialized.
             */
            prepareAndroidNativeLibraries();

            /*
             * ONNX Runtime environment.
             */
            localEnvironment =
                    OrtEnvironment.getEnvironment();

            /*
             * Extract both ONNX files into the same
             * filesystem directory.
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

            System.out.println(
                    "NewGen6: Model directory: " +
                    modelDirectory
            );

            System.out.println(
                    "NewGen6: Model file: " +
                    modelFile
            );

            System.out.println(
                    "NewGen6: External data file: " +
                    modelDataFile
            );

            System.out.println(
                    "NewGen6: Model size: " +
                    Files.size(modelFile) +
                    " bytes"
            );

            System.out.println(
                    "NewGen6: External data size: " +
                    Files.size(modelDataFile) +
                    " bytes"
            );

            /*
             * IMPORTANT:
             *
             * Load from the filesystem path rather than
             * createSession(byte[]).
             *
             * The model uses external tensor data.
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
             * Publish only after the complete session
             * has successfully loaded.
             */
            environment = localEnvironment;
            session = localSession;

            localEnvironment = null;
            localSession = null;

            printModelInterface();

            System.out.println();
            System.out.println(
                    "================================"
            );
            System.out.println(
                    "NewGen6: ONNX MODEL READY"
            );
            System.out.println(
                    "NewGen6: Expected input: [1,96,156]"
            );
            System.out.println(
                    "NewGen6: Features per frame: 78"
            );
            System.out.println(
                    "NewGen6: Model outputs: " +
                    session.getOutputInfo().size()
            );
            System.out.println(
                    "================================"
            );

        } catch (Throwable e) {

            failed = true;

            System.err.println();
            System.err.println(
                    "NewGen6: FAILED TO LOAD ONNX MODEL"
            );
            System.err.println(
                    "NewGen6: Minecraft will NOT be crashed."
            );

            e.printStackTrace();

            if (localSession != null) {
                try {
                    localSession.close();
                } catch (Exception ignored) {
                }
            }

            /*
             * OrtEnvironment is effectively a JVM-wide
             * singleton in current ONNX Runtime versions.
             */
            environment = null;
            session = null;

        } finally {

            loading = false;

            System.out.println(
                    "NewGen6: ONNX loader finished. " +
                    "loaded=" + (session != null) +
                    ", failed=" + failed
            );
        }
    }

    private static void printModelInterface()
            throws OrtException {

        if (session == null) {
            return;
        }

        System.out.println();
        System.out.println(
                "=== NewGen6 ONNX Model Interface ==="
        );

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

        System.out.println(
                "===================================="
        );
    }

    /*
     * Runs one actual model inference.
     *
     * The caller supplies the COMPLETE [1,96,156]
     * tensor contents.
     *
     * This method does NOT construct the 78 features.
     */
    public static synchronized Map<String, float[][]>
    runInference(float[][][] input)
            throws OrtException {

        if (session == null) {
            throw new IllegalStateException(
                    "NewGen6 ONNX session is not loaded."
            );
        }

        validateInputShape(input);

        /*
         * Convert float[96][156] into a batch of one:
         *
         * [1,96,156]
         */
        float[][][] batch =
                new float[1][HISTORY_LENGTH][MODEL_FEATURE_COUNT];

        for (int t = 0; t < HISTORY_LENGTH; t++) {
            System.arraycopy(
                    input[t],
                    0,
                    batch[0][t],
                    0,
                    MODEL_FEATURE_COUNT
            );
        }

        String inputName =
                session.getInputInfo()
                        .keySet()
                        .iterator()
                        .next();

        try (OnnxTensor tensor =
                     OnnxTensor.createTensor(
                             environment,
                             batch
                     )) {

            Map<String, OnnxTensor> inputs =
                    new HashMap<>();

            inputs.put(
                    inputName,
                    tensor
            );

            try (OrtSession.Result result =
                         session.run(inputs)) {

                Map<String, float[][]> outputs =
                        new HashMap<>();

                for (Map.Entry<String, NodeInfo> entry :
                        session.getOutputInfo().entrySet()) {

                    String name = entry.getKey();

                    Object value =
                            result.get(name)
                                  .orElse(null);

                    if (value instanceof float[][]) {

                        outputs.put(
                                name,
                                (float[][]) value
                        );

                    } else if (value instanceof float[]) {

                        float[] flat =
                                (float[]) value;

                        outputs.put(
                                name,
                                new float[][]{
                                        flat
                                }
                        );

                    } else {

                        throw new OrtException(
                                "Unexpected output type for " +
                                name +
                                ": " +
                                (value == null
                                        ? "null"
                                        : value.getClass()
                                                .getName())
                        );
                    }
                }

                return outputs;
            }
        }
    }

    private static void validateInputShape(
            float[][][] input) {

        if (input == null) {
            throw new IllegalArgumentException(
                    "Input tensor is null."
            );
        }

        if (input.length != HISTORY_LENGTH) {
            throw new IllegalArgumentException(
                    "Expected 96 frames, got " +
                    input.length
            );
        }

        for (int t = 0; t < HISTORY_LENGTH; t++) {

            if (input[t] == null) {
                throw new IllegalArgumentException(
                        "Frame " + t + " is null."
                );
            }

            if (input[t].length != MODEL_FEATURE_COUNT) {
                throw new IllegalArgumentException(
                        "Frame " + t +
                        " expected 156 features, got " +
                        input[t].length
                );
            }
        }
    }

    private static synchronized void
    prepareAndroidNativeLibraries()
            throws IOException {

        if (nativeDirectory != null) {

            System.setProperty(
                    "onnxruntime.native.path",
                    nativeDirectory.toString()
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

        extractResource(
                ORT_NATIVE,
                ortPath
        );

        extractResource(
                ORT_JNI_NATIVE,
                jniPath
        );

        /*
         * These sizes are useful for detecting the exact
         * problem you previously had where the resources
         * were only ~100 bytes.
         */
        long ortSize =
                Files.size(ortPath);

        long jniSize =
                Files.size(jniPath);

        System.out.println(
                "NewGen6: libonnxruntime.so size: " +
                ortSize +
                " bytes"
        );

        System.out.println(
                "NewGen6: libonnxruntime4j_jni.so size: " +
                jniSize +
                " bytes"
        );

        if (ortSize < 1_000_000) {

            throw new IOException(
                    "libonnxruntime.so is suspiciously small: " +
                    ortSize +
                    " bytes"
            );
        }

        if (jniSize < 10_000) {

            throw new IOException(
                    "libonnxruntime4j_jni.so is suspiciously small: " +
                    jniSize +
                    " bytes"
            );
        }

        System.setProperty(
                "onnxruntime.native.path",
                directory.toString()
        );

        nativeDirectory = directory;

        System.out.println(
                "NewGen6: Android ARM64 native directory: " +
                directory
        );
    }

    private static synchronized void
    prepareModelFiles()
            throws IOException {

        if (modelDirectory != null) {
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

        extractResource(
                MODEL_PATH,
                modelFile
        );

        extractResource(
                MODEL_DATA_PATH,
                modelDataFile
        );

        if (!Files.exists(modelFile) ||
                Files.size(modelFile) == 0) {

            throw new IOException(
                    "Invalid ONNX model file."
            );
        }

        if (!Files.exists(modelDataFile) ||
                Files.size(modelDataFile) == 0) {

            throw new IOException(
                    "Invalid ONNX external data file."
            );
        }

        modelDirectory = directory;

        System.out.println(
                "NewGen6: Model files extracted successfully."
        );
    }

    private static void extractResource(
            String resourcePath,
            Path destination)
            throws IOException {

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

        OrtSession currentSession =
                session;

        session = null;

        if (currentSession != null) {

            try {
                currentSession.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        /*
         * Do not depend on closing OrtEnvironment to
         * unload native libraries. Current ORT Java
         * documentation describes the environment as
         * effectively JVM-wide.
         */
        environment = null;

        nativeDirectory = null;
        modelDirectory = null;

        System.out.println(
                "NewGen6: ONNX resources closed."
        );
    }
}