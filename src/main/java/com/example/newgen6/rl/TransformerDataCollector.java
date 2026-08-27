package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.zip.GZIPOutputStream;

/**
 * Trajectory collector for offline transformer / imitation learning.
 * Includes episode id + terminal/win/loss flags.
 */
public final class TransformerDataCollector {

    private static final Path DIR = Paths.get("newgen6_transformer_data");
    private static final int MAX_BUFFER = 4000;

    private final Frame[] buf = new Frame[MAX_BUFFER];
    private int size = 0;
    private boolean enabled = true;
    private long totalWritten = 0;
    private long episodeId = 0;
    private int episodeStep = 0;

    public TransformerDataCollector() {
        try { Files.createDirectories(DIR); } catch (IOException ignored) {}
    }

    public void setEnabled(boolean e) { enabled = e; }
    public boolean isEnabled() { return enabled; }
    public long getTotalWritten() { return totalWritten; }
    public int getBufferSize() { return size; }
    public long getEpisodeId() { return episodeId; }

    public void beginEpisode() {
        episodeId++;
        episodeStep = 0;
    }

    public void endEpisode(boolean won, boolean lost) {
        // mark last frame as terminal if present
        if (size > 0 && buf[size - 1] != null) {
            buf[size - 1].terminal = true;
            buf[size - 1].won = won;
            buf[size - 1].lost = lost;
        }
        flush();
        episodeStep = 0;
    }

    public void record(long tick, float[] obs, int action,
                       float mouseDx, float mouseDy, float reward,
                       boolean hit, boolean crit, float dmgDealt, float dmgTaken,
                       boolean hasTarget, float health, float targetHealth) {
        if (!enabled || obs == null) return;
        if (size >= MAX_BUFFER) flush();
        Frame f = new Frame();
        f.tick = tick;
        f.episodeId = episodeId;
        f.episodeStep = episodeStep++;
        f.obs = obs.clone();
        f.action = action;
        f.mouseDx = mouseDx;
        f.mouseDy = mouseDy;
        f.reward = reward;
        f.hit = hit;
        f.crit = crit;
        f.hasTarget = hasTarget;
        f.dmgDealt = dmgDealt;
        f.dmgTaken = dmgTaken;
        f.health = health;
        f.targetHealth = targetHealth;
        f.terminal = false;
        f.won = false;
        f.lost = false;
        buf[size++] = f;
    }

    public synchronized void flush() {
        if (size == 0) return;
        String name = "traj-" + LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
                + "-" + totalWritten + ".bin.gz";
        Path path = DIR.resolve(name);
        try (DataOutputStream out = new DataOutputStream(
                new GZIPOutputStream(new BufferedOutputStream(Files.newOutputStream(path))))) {
            out.writeUTF("NEWGEN6-TRAJ-v2");
            out.writeInt(size);
            out.writeInt(buf[0].obs.length);
            for (int i = 0; i < size; i++) {
                Frame f = buf[i];
                out.writeLong(f.tick);
                out.writeLong(f.episodeId);
                out.writeInt(f.episodeStep);
                for (float v : f.obs) out.writeFloat(v);
                out.writeInt(f.action);
                out.writeFloat(f.mouseDx);
                out.writeFloat(f.mouseDy);
                out.writeFloat(f.reward);
                int flags = (f.hit ? 1 : 0) | (f.crit ? 2 : 0) | (f.hasTarget ? 4 : 0)
                        | (f.terminal ? 8 : 0) | (f.won ? 16 : 0) | (f.lost ? 32 : 0);
                out.writeByte(flags);
                out.writeFloat(f.dmgDealt);
                out.writeFloat(f.dmgTaken);
                out.writeFloat(f.health);
                out.writeFloat(f.targetHealth);
            }
            totalWritten += size;
            System.out.println("[NEWGEN6] Transformer data: " + path.getFileName()
                    + " frames=" + size + " total=" + totalWritten);
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Transformer flush failed: " + e.getMessage());
        }
        size = 0;
    }

    private static final class Frame {
        long tick, episodeId;
        int episodeStep, action;
        float[] obs;
        float mouseDx, mouseDy, reward, dmgDealt, dmgTaken, health, targetHealth;
        boolean hit, crit, hasTarget, terminal, won, lost;
    }
}