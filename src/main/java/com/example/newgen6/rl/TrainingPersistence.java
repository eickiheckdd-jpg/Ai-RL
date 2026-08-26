package com.example.newgen6.rl;

import com.example.newgen6.rl.nn.PolicyValueNetwork;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;

/**
 * Manages the local Java-only model checkpoint.
 *
 * Files are stored under the Minecraft game directory rather than inside
 * the mod JAR, so the learned policy survives rebuilds and mod updates.
 */
public final class TrainingPersistence {

    private static final String DIRECTORY_NAME = "newgen6";
    private static final String CHECKPOINT_NAME =
            "pvp_rl_model.ng6";

    private final Path directory;
    private final Path checkpoint;

    public TrainingPersistence(Path gameDirectory) {
        Objects.requireNonNull(
                gameDirectory,
                "gameDirectory"
        );

        this.directory =
                gameDirectory.resolve(DIRECTORY_NAME);

        this.checkpoint =
                directory.resolve(CHECKPOINT_NAME);
    }

    public Path directory() {
        return directory;
    }

    public Path checkpoint() {
        return checkpoint;
    }

    public boolean exists() {
        return Files.isRegularFile(checkpoint);
    }

    public boolean save(
            PolicyValueNetwork network) {

        try {
            Files.createDirectories(directory);

            ModelCheckpoint.save(
                    checkpoint,
                    network
            );

            return true;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    public boolean load(
            PolicyValueNetwork network) {

        if (!exists()) {
            return false;
        }

        try {
            ModelCheckpoint.load(
                    checkpoint,
                    network
            );

            return true;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Moves a known-good checkpoint aside rather than destroying it.
     */
    public boolean quarantineBadCheckpoint() {

        if (!exists()) {
            return false;
        }

        try {
            Path backup = directory.resolve(
                    CHECKPOINT_NAME + ".bad"
            );

            Files.move(
                    checkpoint,
                    backup,
                    StandardCopyOption.REPLACE_EXISTING
            );

            return true;
        } catch (IOException e) {
            return false;
        }
    }
}