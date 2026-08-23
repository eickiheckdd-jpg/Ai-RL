package com.example.newgen6;

import net.fabricmc.loader.api.FabricLoader;
import java.io.File;

public class RLConfig {
    public static boolean agentEnabled = false;
    public static boolean showDebugHud = true;
    public static final File MODEL_FILE = FabricLoader.getInstance().getConfigDir().resolve("newgen6_ppo.zip").toFile();

    // Environment Dimensions
    public static final int OBS_DIM = 6;             // dx, dy, dz, pitch, yaw, dist
    public static final int ACTION_CONTINUOUS_DIM = 2; // yaw_delta, pitch_delta
    public static final float MAX_YAW_DELTA = 10.0f;
    public static final float MAX_PITCH_DELTA = 8.0f;

    // Ultra-Low Mobile Resource Settings
    public static final float LEARNING_RATE = 1e-3f;
    public static final int N_STEPS = 128;           // Tiny step buffer
    public static final int BATCH_SIZE = 16;
    public static final int N_EPOCHS = 1;            // Only 1 training pass per cycle
    public static final int TICK_INTERVAL = 4;       // Run AI every 4 ticks (~5 times/sec)
}
