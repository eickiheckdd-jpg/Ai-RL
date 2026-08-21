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

    private static volatile boolean inferenceProbeStarted = false;
    private static volatile boolean inferenceProbeFinished = false;

    private static volatile Path nativeDirectory;
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

            prepareAndroidNativeLibraries();

            localEnvironment =
                    OrtEnvironment.getEnvironment();

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

            environment = localEnvironment;
            session = localSession;

            localEnvironment = null;
            localSession = null;

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
             * Controlled zero-input inference test.
             * This proves the ONNX execution path works.
             * It is NOT the real Minecraft feature vector.
             */
            runInferenceProbe();

            /*
             * Existing interface diagnostic.
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

        extractResource(
                ORT_NATIVE,
                ortPath
        );

        extractResource(
                ORT_JNI_NATIVE,
                jniPath
        );

        System.setProperty(
                "onnxruntime.native.path",
                directory.toString()
        );

        nativeDirectory = directory;

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

        extractResource(
                MODEL_PATH,
                modelFile
        );

        extractResource(
                MODEL_DATA_PATH,
                modelDataFile
        );

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
     * Controlled ONNX execution test.
     *
     * Uses zeros only to verify that session.run() works.
     * It is NOT the real Minecraft AI input.
     */

    /**
     * Result returned by a real feature-tensor inference call.
     * The tensor must have shape [1,96,156].
     */
    public static final class InferenceResult {

        private final String[] outputNames;
        private final Object[] outputValues;

        private InferenceResult(
                String[] outputNames,
                Object[] outputValues
        ) {
            this.outputNames = outputNames;
            this.outputValues = outputValues;
        }

        public String[] getOutputNames() {
            return outputNames.clone();
        }

        public Object[] getOutputValues() {
            return outputValues.clone();
        }

        public Object getOutput(String name) {
            for (int i = 0; i < outputNames.length; i++) {
                if (outputNames[i].equals(name)) {
                    return outputValues[i];
                }
            }
            return null;
        }

        public float getScalar(String name) {
            Object value = getOutput(name);

            if (value instanceof float[][]) {
                float[][] array = (float[][]) value;
                if (array.length > 0 && array[0].length > 0) {
                    return array[0][0];
                }
            }

            if (value instanceof float[]) {
                float[] array = (float[]) value;
                if (array.length > 0) {
                    return array[0];
                }
            }

            throw new IllegalStateException(
                    "Output '" + name +
                    "' is not a scalar float output."
            );
        }
    }

    /**
     * Runs the ONNX model using a real gameplay feature tensor.
     *
     * Required tensor shape: [1,96,156].
     *
     * FeatureTensor.java is responsible for constructing the tensor.
     * This method does NOT create a zero tensor.
     */
    public static InferenceResult runInference(
            float[][][] featureTensor
    ) {
        if (session == null || environment == null) {
            throw new IllegalStateException(
                    "NewGen6: ONNX model is not loaded."
            );
        }

        validateFeatureTensor(featureTensor);

        try (
                OnnxTensor inputTensor =
                        OnnxTensor.createTensor(
                                environment,
                                featureTensor
                        )
        ) {
            try (
                    OrtSession.Result results =
                            session.run(
                                    Collections.singletonMap(
                                            "input",
                                            inputTensor
                                    )
                            )
            ) {
                int outputCount = results.size();

                if (outputCount != 29) {
                    throw new IllegalStateException(
                            "NewGen6: Expected 29 outputs, " +
                            "received " + outputCount
                    );
                }

                String[] outputNames =
                        session.getOutputNames()
                               .toArray(new String[0]);

                Object[] outputValues =
                        new Object[outputCount];

                System.out.println(
                        "NewGen6: REAL INFERENCE completed"
                );

                for (int i = 0; i < outputCount; i++) {
                    String outputName =
                            i < outputNames.length
                                    ? outputNames[i]
                                    : "output_" + i;

                    Object value =
                            results.get(i).getValue();

                    outputValues[i] = value;

                    System.out.println(
                            "NewGen6: REAL " +
                            outputName +
                            " = " +
                            formatOutput(value)
                    );
                }

                return new InferenceResult(
                        outputNames,
                        outputValues
                );
            }
        } catch (Throwable e) {
            System.err.println(
                    "NewGen6: REAL INFERENCE FAILED"
            );
            e.printStackTrace();

            throw new RuntimeException(
                    "NewGen6: ONNX real inference failed.",
                    e
            );
        }
    }

    /**
     * Validates the exact tensor dimensions expected by the model.
     */
    private static void validateFeatureTensor(
            float[][][] featureTensor
    ) {
        if (featureTensor == null) {
            throw new IllegalArgumentException(
                    "NewGen6: Feature tensor is null."
            );
        }

        if (featureTensor.length != 1) {
            throw new IllegalArgumentException(
                    "NewGen6: Feature tensor batch dimension " +
                    "must be 1, got " + featureTensor.length
            );
        }

        if (featureTensor[0] == null ||
            featureTensor[0].length != 96) {

            int frames =
                    featureTensor[0] == null
                            ? -1
                            : featureTensor[0].length;

            throw new IllegalArgumentException(
                    "NewGen6: Feature tensor must contain " +
                    "96 frames, got " + frames
            );
        }

        for (int frame = 0; frame < 96; frame++) {
            if (featureTensor[0][frame] == null ||
                featureTensor[0][frame].length != 156) {

                int features =
                        featureTensor[0][frame] == null
                                ? -1
                                : featureTensor[0][frame].length;

                throw new IllegalArgumentException(
                        "NewGen6: Frame " + frame +
                        " must contain 156 values, got " +
                        features
                );
            }

            for (int feature = 0; feature < 156; feature++) {
                float value =
                        featureTensor[0][frame][feature];

                if (Float.isNaN(value) ||
                    Float.isInfinite(value)) {

                    throw new IllegalArgumentException(
                            "NewGen6: Invalid feature value at " +
                            "[0][" + frame + "][" +
                            feature + "]: " + value
                    );
                }
            }
        }
    }

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

            float[][][] inputData =
                    new float[1][96][156];

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
                            "NewGen6: Output count = " +
                            outputCount
                    );

                    String[] outputNames =
                            session.getOutputNames()
                                   .toArray(
                                           new String[0]
                                   );

                    for (int i = 0;
                         i < outputCount;
                         i++) {

                        String outputName =
                                i < outputNames.length
                                        ? outputNames[i]
                                        : "output_" + i;

                        Object value =
                                results.get(i).getValue();

                        System.out.println(
                                "NewGen6: " +
                                outputName +
                                " = " +
                                formatOutput(value)
                        );
                    }

                    if (outputCount == 29) {

                        System.out.println();
                        System.out.println(
                                "NewGen6: 29/29 outputs received"
                        );
                        System.out.println(
                                "NewGen6: INFERENCE PROBE PASSED"
                        );

                    } else {

                        System.err.println();
                        System.err.println(
                                "NewGen6: INFERENCE PROBE WARNING"
                        );
                        System.err.println(
                                "NewGen6: Expected 29 outputs, " +
                                "received " +
                                outputCount
                        );
                    }
                }
            }

            inferenceProbeFinished = true;

        } catch (Throwable e) {

            System.err.println();
            System.err.println(
                    "NewGen6: INFERENCE PROBE FAILED"
            );

            e.printStackTrace();

            inferenceProbeFinished = true;
        }

        System.out.println();
        System.out.println(
                "================================"
        );
        System.out.println(
                "NewGen6: INFERENCE PROBE END"
        );
        System.out.println(
                "================================"
        );
    }

    private static String formatOutput(Object value) {

        if (value == null) {
            return "null";
        }

        if (value instanceof float[]) {
            return Arrays.toString(
                    (float[]) value
            );
        }

        if (value instanceof float[][]) {
            return Arrays.deepToString(
                    (float[][]) value
            );
        }

        if (value instanceof float[][][]) {
            return Arrays.deepToString(
                    (float[][][]) value
            );
        }

        if (value instanceof double[]) {
            return Arrays.toString(
                    (double[]) value
            );
        }

        if (value instanceof long[]) {
            return Arrays.toString(
                    (long[]) value
            );
        }

        if (value instanceof int[]) {
            return Arrays.toString(
                    (int[]) value
            );
        }

        return String.valueOf(value);
    }

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
                "NewGen6: MODEL INTERFACE TEST"
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

            System.out.println();
            System.out.println(
                    "NewGen6: Model interface verified."
            );

            inferenceTestFinished = true;

        } catch (Throwable e) {

            System.err.println(
                    "NewGen6: MODEL INTERFACE TEST FAILED"
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

    public static boolean isInferenceProbeFinished() {
        return inferenceProbeFinished;
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

        inferenceProbeStarted = false;
        inferenceProbeFinished = false;
        inferenceTestStarted = false;
        inferenceTestFinished = false;

        System.out.println(
                "NewGen6: ONNX resources closed."
        );
    }
}
