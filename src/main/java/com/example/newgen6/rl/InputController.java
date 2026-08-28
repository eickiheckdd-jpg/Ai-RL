package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import org.lwjgl.glfw.GLFW;

public final class InputController {
    private final double[] cursorX = new double[1];
    private final double[] cursorY = new double[1];

    private boolean controlling = false;

    public void apply(MinecraftClient client, ActionSample action) {
        if (client == null || client.player == null || client.world == null || client.options == null) {
            releaseAll(client);
            return;
        }

        if (client.currentScreen != null || client.isPaused()) {
            releaseAll(client);
            return;
        }

        controlling = true;

        setKey(client.options.forwardKey, action.forward);
        setKey(client.options.backKey, action.backward);
        setKey(client.options.leftKey, action.left);
        setKey(client.options.rightKey, action.right);

        setKey(client.options.jumpKey, action.jump);
        setKey(client.options.sprintKey, action.sprint);
        setKey(client.options.sneakKey, action.sneak);
        setKey(client.options.attackKey, action.attack);

        applyMouseDelta(client, action.mouseX, action.mouseY);
    }

    public void releaseAll(MinecraftClient client) {
        if (client == null || client.options == null) {
            controlling = false;
            return;
        }

        setKey(client.options.forwardKey, false);
        setKey(client.options.backKey, false);
        setKey(client.options.leftKey, false);
        setKey(client.options.rightKey, false);

        setKey(client.options.jumpKey, false);
        setKey(client.options.sprintKey, false);
        setKey(client.options.sneakKey, false);
        setKey(client.options.attackKey, false);

        controlling = false;
    }

    public boolean isControlling() {
        return controlling;
    }

    private void setKey(KeyBinding key, boolean pressed) {
        if (key != null) {
            key.setPressed(pressed);
        }
    }

    private void applyMouseDelta(MinecraftClient client, float dx, float dy) {
        try {
            dx = clamp(dx, -AgentConfig.MOUSE_CLAMP, AgentConfig.MOUSE_CLAMP);
            dy = clamp(dy, -AgentConfig.MOUSE_CLAMP, AgentConfig.MOUSE_CLAMP);

            if (dx == 0.0f && dy == 0.0f) return;

            long window = client.getWindow().getHandle();
            if (window == 0L) return;

            GLFW.glfwGetCursorPos(window, cursorX, cursorY);
            GLFW.glfwSetCursorPos(window, cursorX[0] + dx, cursorY[0] + dy);
        } catch (Throwable ignored) {
            // If platform/GLFW does not support synthetic cursor movement, do not crash.
        }
    }

    private static float clamp(float v, float min, float max) {
        if (!Float.isFinite(v)) return min;
        return Math.max(min, Math.min(max, v));
    }
}