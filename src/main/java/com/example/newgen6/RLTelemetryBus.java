package com.example.newgen6;

import java.util.concurrent.atomic.AtomicReference;

public class RLTelemetryBus {
    public record TelemetrySnapshot(
        float[] observation,
        float yawDelta,
        float pitchDelta,
        float valueEstimate,
        float lastReward,
        float totalEpisodeReward,
        int episodeLength
    ) {}

    private static final AtomicReference<TelemetrySnapshot> CURRENT_SNAPSHOT = 
        new AtomicReference<>(new TelemetrySnapshot(new float[RLConfig.OBS_DIM], 0, 0, 0, 0, 0, 0));

    public static void update(TelemetrySnapshot snapshot) {
        CURRENT_SNAPSHOT.set(snapshot);
    }

    public static TelemetrySnapshot get() {
        return CURRENT_SNAPSHOT.get();
    }
}
