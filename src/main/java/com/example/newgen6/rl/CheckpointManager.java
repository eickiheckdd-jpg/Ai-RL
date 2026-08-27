package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Persistent model saving / loading.
 * Saves network weights + training state so a crash does not lose everything.
 * Keeps both "latest" and "best" checkpoints separately.
 */
public final class CheckpointManager {

    private static final Path DIR = Paths.get("newgen6_checkpoints");
    private static final String LATEST = "latest.bin";
    private static final String BEST = "best.bin";

    private float bestScore = Float.NEGATIVE_INFINITY;

    public CheckpointManager() {
        try {
            Files.createDirectories(DIR);
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Could not create checkpoint dir: " + e.getMessage());
        }
    }

    /** Save full agent state (weights + curriculum + stats) */
    public void save(PPOAgent agent, String name) {
        Path path = DIR.resolve(name);
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {

            // header
            out.writeUTF("NEWGEN6-CKPT-v1");
            out.writeLong(agent.getGlobalSteps());
            out.writeInt(agent.getCurriculum().getStage());
            out.writeFloat(agent.getCurriculum().getNoiseScale());
            out.writeFloat(agent.getCurriculum().getLookScale());
            out.writeBoolean(agent.isTraining());

            // network weights (simplified – real version would dump all matrices)
            agent.getNetwork().serialize(out);

            out.flush();
            System.out.println("[NEWGEN6] Checkpoint saved → " + path.toAbsolutePath());
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Save failed: " + e.getMessage());
        }
    }

    public void saveLatest(PPOAgent agent) {
        save(agent, LATEST);
    }

    public void saveBest(PPOAgent agent, float score) {
        if (score > bestScore) {
            bestScore = score;
            save(agent, BEST);
            System.out.println("[NEWGEN6] ★ New best model saved (score=" + score + ")");
        }
    }

    public boolean loadLatest(PPOAgent agent) {
        return load(agent, LATEST);
    }

    public boolean loadBest(PPOAgent agent) {
        return load(agent, BEST);
    }

    private boolean load(PPOAgent agent, String name) {
        Path path = DIR.resolve(name);
        if (!Files.exists(path)) {
            System.out.println("[NEWGEN6] No checkpoint found: " + name);
            return false;
        }
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {

            String magic = in.readUTF();
            if (!magic.startsWith("NEWGEN6-CKPT")) {
                System.err.println("[NEWGEN6] Invalid checkpoint format");
                return false;
            }
            long steps = in.readLong();
            int stage = in.readInt();
            float noise = in.readFloat();
            float look = in.readFloat();
            boolean training = in.readBoolean();

            agent.getNetwork().deserialize(in);
            agent.restoreState(steps, stage, noise, look, training);

            System.out.println("[NEWGEN6] Loaded " + name + " (steps=" + steps + ", stage=" + stage + ")");
            return true;
        } catch (IOException e) {
            System.err.println("[NEWGEN6] Load failed: " + e.getMessage());
            return false;
        }
    }

    /** Timestamped snapshot for versioning */
    public void saveVersioned(PPOAgent agent) {
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        save(agent, "gen6-" + ts + ".bin");
    }
}
