package com.example.newgen6.rl.env;

import com.example.newgen6.mixin.PvPMixin;

public final class PvPStateStore {

    private static volatile PvPMixin.Snapshot latestSnapshot =
            PvPMixin.Snapshot.empty();

    private PvPStateStore() {
        // Utility class; no instances.
    }

    public static PvPMixin.Snapshot getLatestSnapshot() {
        return latestSnapshot;
    }

    public static void setLatestSnapshot(
            PvPMixin.Snapshot snapshot) {

        if (snapshot != null) {
            latestSnapshot = snapshot;
        }
    }
}