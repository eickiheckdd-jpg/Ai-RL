package com.example.newgen6.rl.env;

public final class PvPStateStore {

    private static volatile PvPSnapshot latestSnapshot =
            PvPSnapshot.empty();

    private PvPStateStore() {
    }

    public static PvPSnapshot getLatestSnapshot() {
        return latestSnapshot;
    }

    public static void setLatestSnapshot(
            PvPSnapshot snapshot) {

        if (snapshot != null) {
            latestSnapshot = snapshot;
        }
    }
}