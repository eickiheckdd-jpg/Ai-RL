package com.example.newgen6.rl;

import com.example.newgen6.NewGen6RLMod;
import net.fabricmc.loader.api.FabricLoader;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

public final class ModelCheckpoint {
    private static final int MAGIC = 0x4E473643; // "NG6C"

    private ModelCheckpoint() {
    }

    public static Path latestPath() {
        return FabricLoader.getInstance()
                .getConfigDir()
                .resolve("newgen6")
                .resolve("latest.ng6");
    }

    public static void save(PpoAgent agent, Path path, int episode, long globalStep) throws IOException {
        if (agent == null || path == null) return;

        Files.createDirectories(path.getParent());
        Path tmp = path.resolveSibling(path.getFileName().toString() + ".tmp");

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(
                        Files.newOutputStream(tmp, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
                )
        )) {
            out.writeInt(MAGIC);
            out.writeInt(AgentConfig.CHECKPOINT_VERSION);
            out.writeLong(System.currentTimeMillis());
            out.writeInt(episode);
            out.writeLong(globalStep);

            agent.writeModel(out);
        }

        try {
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
        }

        NewGen6RLMod.LOGGER.info("[NEWGEN6] Checkpoint saved: {}", path);
    }

    public static boolean load(PpoAgent agent, Path path) {
        if (agent == null || path == null || !Files.exists(path)) {
            return false;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            int version = in.readInt();

            if (magic != MAGIC) {
                NewGen6RLMod.LOGGER.warn("[NEWGEN6] Checkpoint magic mismatch");
                return false;
            }

            if (version != AgentConfig.CHECKPOINT_VERSION) {
                NewGen6RLMod.LOGGER.warn(
                        "[NEWGEN6] Checkpoint version mismatch: file={} expected={}",
                        version,
                        AgentConfig.CHECKPOINT_VERSION
                );
                return false;
            }

            in.readLong();  // timestamp
            in.readInt();   // episode
            in.readLong();  // global step

            boolean ok = agent.readModel(in);
            if (ok) {
                NewGen6RLMod.LOGGER.info("[NEWGEN6] Checkpoint loaded: {}", path);
            } else {
                NewGen6RLMod.LOGGER.warn("[NEWGEN6] Failed to apply checkpoint contents: {}", path);
            }
            return ok;
        } catch (Exception e) {
            NewGen6RLMod.LOGGER.error("[NEWGEN6] Failed to load checkpoint", e);
            return false;
        }
    }
}