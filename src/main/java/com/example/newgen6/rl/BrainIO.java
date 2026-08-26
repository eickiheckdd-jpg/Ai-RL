package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;

import java.nio.file.Path;

/**
 * Default brain path: {@code .minecraft/config/newgen6/brain.ng6}
 */
public final class BrainIO {
    private BrainIO() {}

    public static final String FILE_NAME = "brain.ng6";

    public static Path defaultPath() {
        MinecraftClient mc = MinecraftClient.getInstance();
        Path root = mc != null && mc.runDirectory != null
                ? mc.runDirectory.toPath()
                : Path.of(".");
        return root.resolve("config").resolve("newgen6").resolve(FILE_NAME);
    }
}