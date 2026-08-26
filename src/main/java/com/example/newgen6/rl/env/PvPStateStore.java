package com.example.newgen6.rl.env;

import com.example.newgen6.mixin.PvPMixin;

public final class PvPStateStore {

    private static volatile PvPSnapshot latestSnapshot =
            PvPSnapshot.empty();

    private PvPStateStore() {
    }

    public static PvPSnapshot getLatestSnapshot() {
        return latestSnapshot;
    }

    public static void setLatestSnapshot(
            PvPMixin.Snapshot snapshot) {

        if (snapshot != null) {
            latestSnapshot = snapshot;
        }
    }
}