package com.example.newgen6.client;

import com.example.newgen6.rl.Action;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;

/**
 * Maps policy actions to Minecraft's normal client key bindings.
 *
 * No fixed combat cooldown or timing is introduced here.
 */
public final class HumanInputBridge {

    private HumanInputBridge() {
    }

    public static void apply(MinecraftClient client, Action action) {
        if (client == null || client.options == null || action == null) {
            return;
        }

        setPressed(client.options.forwardKey, false);
        setPressed(client.options.backKey, false);
        setPressed(client.options.leftKey, false);
        setPressed(client.options.rightKey, false);

        switch (action.move()) {
            case 1 -> setPressed(client.options.forwardKey, true);
            case 2 -> setPressed(client.options.backKey, true);
            case 3 -> setPressed(client.options.leftKey, true);
            case 4 -> setPressed(client.options.rightKey, true);
            case 5 -> {
                setPressed(client.options.forwardKey, true);
                setPressed(client.options.leftKey, true);
            }
            case 6 -> {
                setPressed(client.options.forwardKey, true);
                setPressed(client.options.rightKey, true);
            }
            case 7 -> {
                setPressed(client.options.backKey, true);
                setPressed(client.options.leftKey, true);
            }
            case 8 -> {
                setPressed(client.options.backKey, true);
                setPressed(client.options.rightKey, true);
            }
            default -> {
                // 0 = neutral.
            }
        }

        setPressed(client.options.jumpKey, action.jump() != 0);
        setPressed(client.options.sprintKey, action.sprint() != 0);
        setPressed(client.options.sneakKey, action.sneak() != 0);
        setPressed(client.options.attackKey, action.attack() != 0);
    }

    public static void releaseAll(MinecraftClient client) {
        if (client == null || client.options == null) {
            return;
        }

        setPressed(client.options.forwardKey, false);
        setPressed(client.options.backKey, false);
        setPressed(client.options.leftKey, false);
        setPressed(client.options.rightKey, false);
        setPressed(client.options.jumpKey, false);
        setPressed(client.options.sprintKey, false);
        setPressed(client.options.sneakKey, false);
        setPressed(client.options.attackKey, false);
    }

    private static void setPressed(KeyBinding key, boolean pressed) {
        if (key != null) {
            key.setPressed(pressed);
        }
    }
}