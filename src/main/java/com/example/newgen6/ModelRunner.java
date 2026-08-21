package com.example.newgen6;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class ModelRunner {

    /*
     * ============================================================
     * MODEL RESOURCES
     * ============================================================
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
     * ============================================================
     * BACKGROUND LOADER
     * ============================================================
     */

    private static final ExecutorService LOADER =
            Executors.newSingleThreadExecutor(r -> {

                Thread thread =
                        new Thread(
                                r,
                                "NewGen6-ONNX-Loader"
                        );

                thread.setDaemon(true);

                return thread;
            });


    /*
     * ============================================================
     * ONNX STATE
     * ============================================================
     */

    private static volatile OrtEnvironment environment;

    private static volatile OrtSession session;

    private static volatile boolean loading = false;

    private static volatile boolean failed = false;

    private static volatile boolean inferenceTestStarted = false;

    private static volatile boolean inferenceTestFinished = false;

    private static volatile boolean inferenceProbeStarted = false;

    private static volatile boolean inferenceProbeFinished = false;


    /*
     * ============================================================
     * EXTRACTED DIRECTORIES
     * ============================================================
     */

    private static volatile Path nativeDirectory;

    private static volatile Path modelDirectory;


    private ModelRunner() {
    }


    /*
     * ============================================================
     * PUBLIC INITIALIZATION
     * ============================================================
     */

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

        LOADER.execute(
                ModelRunner::loadModel
        );
    }


    /*
     * ============================================================
     * MODEL LOADING
     * ============================================================
     */

    private static void loadModel() {

        OrtEnvironment localEnvironment = null;

        OrtSession localSession = null;

        try {

            System.out.println(
                    "================================"
            );

            System.out.println(
                    "NewGen6: Loading ONNX model..."
            );

            System.out.println(
                    "Thread: " +
                    Thread.currentThread().getName()
            );

            System.out.println(
                    "================================"
            );


            /*
             * ----------------------------------------------------
             * 1. Prepare native libraries.
             * ----------------------------------------------------
             */

            prepareAndroidNativeLibraries();


            /*
             * ----------------------------------------------------
             * 2. Initialize ONNX Runtime.
             * ----------------------------------------------------
             */

            localEnvironment =
                    OrtEnvironment.getEnvironment();


            /*
             * ----------------------------------------------------
             * 3. Extract model + external data.
             * ----------------------------------------------------
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
             * ----------------------------------------------------
             * 4. Create ONNX session.
             * ----------------------------------------------------
             *
             * IMPORTANT:
             *
             * The model uses external tensor data, therefore
             * we load it from its filesystem path.
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
             * ----------------------------------------------------
             * 5. Publish successfully created session.
             * ----------------------------------------------------
             */

            environment =
                    localEnvironment;

            session =
                    localSession;

            localEnvironment = null;

            localSession = null;


            /*
             * ----------------------------------------------------
             * 6. Print model interface.
             * ----------------------------------------------------
             */

            System.out.println();

            System.out.println(
                    "=== NewGen6 ONNX Model Interface ==="
            );


            System.out.println();

            System.out.println(
                    "INPUTS:"
            );

            for (
                    Map.Entry<String, NodeInfo> entry :
                    session.getInputInfo().entrySet()
            ) {

                System.out.println(
                        "  " +
                        entry.getKey() +
                        " -> " +
                        entry.getValue().getInfo()
                );
            }


            System.out.println();

            System.out.println(
                    "OUTPUTS:"
            );

            for (
                    Map.Entry<String, NodeInfo> entry :
                    session.getOutputInfo().entrySet()
            ) {

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


            /*
             * ----------------------------------------------------
             * 7. Model ready.
             * ----------------------------------------------------
             */

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


            /*
             * ----------------------------------------------------
             * 8. Run controlled inference probe.
             * ----------------------------------------------------
             *
             * This does NOT use Minecraft features.
             *
             * It only proves that:
             *
             * Java
             *   -> ONNX tensor
             *   -> ONNX Runtime
             *   -> model
             *   -> outputs
             *
             * all work correctly.
             */

            runInferenceProbe();


            /*
             * ----------------------------------------------------
             * 9. Existing interface diagnostic.
             * ----------------------------------------------------
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
             * Clean up partially-created session.
             */

            if (localSession != null) {

                try {

                    localSession.close();

                } catch (Exception ignored) {
                }
            }


            /*
             * Clean up partially-created environment.
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
                    "loaded=" +
                    (session != null) +
                    ", failed=" +
                    failed
            );
        }
    }


    /*
     * ============================================================
     * ANDROID NATIVE LIBRARY EXTRACTION
     * ============================================================
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
         * Extract REAL native libraries.
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
         * Tell ONNX Runtime where they are.
         */

        System.setProperty(
                "onnxruntime.native.path",
                directory.toString()
        );


        nativeDirectory =
                directory;


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

        System.out.println(
                "NewGen6: Android ARM64 native directory: " +
                directory
        );
    }


    /*
     * ============================================================
     * MODEL EXTRACTION
     * ============================================================
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
         * Extract ONNX model.
         */

        extractResource(
                MODEL_PATH,
                modelFile
        );


        /*
         * Extract external tensor data.
         *
         * The filename MUST remain exactly:
         *
         * newgen6_full.onnx.data
         */

        extractResource(
                MODEL_DATA_PATH,
                modelDataFile
        );


        /*
         * Verify files.
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


        modelDirectory =
                directory;


        System.out.println(
                "NewGen6: Model files extracted successfully."
        );
    }


    /*
     * ============================================================
     * RESOURCE EXTRACTION
     * ============================================================
     */

    private static void extractResource(
            String resourcePath,
            Path destination
    ) throws IOException {

        try (
                InputStream input =
                        ModelRunner.class
                                .getResourceAsStream(
                                        resourcePath
                                )
        ) {

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


    /*
     * ============================================================
     * REAL ONNX INFERENCE PROBE
     * ============================================================
     *
     * This is deliberately NOT the real Minecraft AI yet.
     *
     * It feeds a zero-filled [1,96,156] tensor only to verify
     * that ONNX Runtime can execute the exported model and return
     * all 29 outputs on the Android device.
     */

    private static void runInferenceProbe() {

        if (session == null ||
            environment == null) {

            System.err.println(
                    "NewGen6: Cannot run inference probe " +
                    "because the model is not loaded."
            );

            return;
        }


        if (inferenceProbeStarted) {

            return;
        }


        inferenceProbeStarted = true;


        System.out.println();

        System.out.println(
                "================================"
        );

        System.out.println(
                "NewGen6: INFERENCE PROBE START"
        );

        System.out.println(
                "================================"
        );


        try {

            /*
             * Exact model input shape for batch size 1.
             *
             * [batch][history][features]
             *
             * [1][96][156]
             */

            float[][][] inputData =
                    new float[1][96][156];


            /*
             * Java arrays are initialized to zero.
             *
             * Therefore this is a completely deterministic
             * zero-input probe.
             */

            System.out.println(
                    "NewGen6: Creating zero tensor..."
            );


            try (
                    OnnxTensor inputTensor =
                            OnnxTensor.createTensor(
                                    environment,
                                    inputData
                            )
            ) {

                System.out.println(
                        "NewGen6: Input tensor = [1,96,156]"
                );


                /*
                 * ------------------------------------------------
                 * Run the actual ONNX model.
                 * ------------------------------------------------
                 */

                try (
                        OrtSession.Result results =
                                session.run(
                                        Collections.singletonMap(
                                                "input",
                                                inputTensor
                                        )
                                )
                ) {

                    System.out.println(
                            "NewGen6: session.run() completed"
                    );


                    int outputCount =
                            results.size();


                    System.out.println(
                