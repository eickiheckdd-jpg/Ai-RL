package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Records states, actions, rewards and damage events for later debugging.
 * Can be turned on/off at runtime.
 */
public final class ReplayRecorder {

    private static final Path DIR = Paths.get("newgen6_replays");
    private boolean recording = false;
    private final List<Frame> frames = new ArrayList<>(2048);

    public static final class Frame {
        public final long tick;
        public final float[] obs;
        public final int action;
        public final float reward;
        public final float dmgDealt, dmgTaken;
        public final boolean hit, crit;
        public final String note;

        public Frame(long tick, float[] obs, int action, float reward,
                     float dmgDealt, float dmgTaken, boolean hit, boolean crit, String note) {
            this.tick = tick;
            this.obs = obs.clone();
            this.action = action;
            this.reward = reward;
            this.dmgDealt = dmgDealt;
            this.dmgTaken = dmgTaken;
            this.hit = hit;
            this.crit = crit;
            this.note = note;
        }
    }

    public ReplayRecorder() {
        try { Files.createDirectories(DIR); } catch (IOException ignored) {}
    }

    public void start() {
        frames.clear();
        recording = true;
        System.out.println("[NEWGEN6] Replay recording STARTED");
    }

    public void stopAndSave() {
        if (!recording) return;
        recording = false;
        String name = "replay-" + LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")) + ".bin";
        Path path = DIR.resolve(name);
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(frames.size());
            for (Frame f : frames) {
                out.writeLong(f.tick);
                out.writeInt(f.obs.length);
                for (float v : f.obs) out.writeFloat(v);
                out.writeInt(f.action);
                out.writeFloat(f.reward);
                out.writeFloat(f.dmgDealt);
                out.writeFloat(f.dmgTaken);
                out.writeBoolean(f.hit);
                out.writeBoolean(f.crit);
                out.writeUTF(f.note == null ? "" : f.note);
            }
            System.out.println("[NEWGEN6] Replay saved → " + path + " (" + frames.size() + " frames)");
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Replay save failed: " + e.getMessage());
        }
        frames.clear();
    }

    public void record(long tick, float[] obs, int action, float reward,
                       float dmgDealt, float dmgTaken, boolean hit, boolean crit, String note) {
        if (!recording) return;
        if (frames.size() > 8000) return; // safety
        frames.add(new Frame(tick, obs, action, reward, dmgDealt, dmgTaken, hit, crit, note));
    }

    public boolean isRecording() { return recording; }
    public int size() { return frames.size(); }
}
