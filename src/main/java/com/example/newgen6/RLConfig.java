package com.example.newgen6;

import net.fabricmc.loader.api.FabricLoader;
import java.io.File;

public class RLConfig {
    public static boolean agentEnabled = false;
    public static boolean showDebugHud = true;
    public static final File MODEL_FILE = FabricLoader.getInstance().getConfigDir().resolve("newgen6_ppo_brain.zip").toFile();

    public static final int OBS_DIM = 14;
    public static final int ACTION_CONTINUOUS_DIM = 2; // yaw_delta, pitch_delta
    public static final float MAX_YAW_DELTA = 15.0f;
    public static final float MAX_PITCH_DELTA = 10.0f;
    
    // Mobile optimizations
    public static final float LEARNING_RATE = 5e-4f;
    public static final int N_STEPS = 512; 
    public static final int BATCH_SIZE = 32; 
    public static final int N_EPOCHS = 3; 
    public static final int TICK_INTERVAL = 2; // Run AI every 2 ticks
}
