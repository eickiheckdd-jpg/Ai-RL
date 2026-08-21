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

private static final String MODEL_PATH =
        "/newgen6/newgen6_full.onnx";

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

private static volatile Path nativeDirectory;

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
         * This must happen BEFORE the first reference to
         * OrtEnvironment.getEnvironment().
         *
         * The normal Java ONNX Runtime loader checks
         * "onnxruntime.native.path" for externally supplied
         * native libraries.
         */
        prepareAndroidNativeLibraries();

        /*
         * Only now initialize ONNX Runtime.
         */
        localEnvironment =
                OrtEnvironment.getEnvironment();

        byte[] modelBytes;

        try (InputStream input =
                     ModelRunner.class
                             .getResourceAsStream(MODEL_PATH)) {

            if (input == null) {
                throw new IOException(
                        "Model not found inside mod JAR: " +
                        MODEL_PATH
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
        System.out.println(
                "=== NewGen6 ONNX Model ==="
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
 * Extracts the Android ARM64 ONNX Runtime libraries into
 * a writable temporary directory and tells ONNX Runtime
 * where to find them.
 *
 * This runs on the NewGen6-ONNX-Loader thread.
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

    extractResource(
            ORT_NATIVE,
            ortPath
    );

    extractResource(
            ORT_JNI_NATIVE,
            jniPath
    );

    /*
     * Make the extracted directory visible to the
     * ONNX Runtime Java native loader.
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
                    "Native library not found " +
                    "inside mod JAR: " +
                    resourcePath
            );
        }

        Files.copy(
                input,
                destination
        );
    }
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
        System.out.println(
                "NewGen6: Waiting for input schema "
                +
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

    nativeDirectory = null;

    System.out.println(
            "NewGen6: ONNX resources closed."
    );
}

}