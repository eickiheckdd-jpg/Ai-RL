package com.example.newgen6;

import net.minecraft.client.MinecraftClient;

public enum CombatAction {
    ATTACK {
        @Override
        public void execute(MinecraftClient client) {
            if (client.options != null) client.options.attackKey.setPressed(true);
        }
    },
    JUMP {
        @Override
        public void execute(MinecraftClient client) {
            if (client.options != null) client.options.jumpKey.setPressed(true);
        }
    },
    MOVE_FORWARD {
        @Override
        public void execute(MinecraftClient client) {
            if (client.options != null) {
                client.options.forwardKey.setPressed(true);
                // Optional: let it learn to sprint if it holds forward while fed
                if (client.player != null && !client.player.isSprinting() && client.player.getHungerManager().getFoodLevel() > 6) {
                    client.player.setSprinting(true);
                }
            }
        }
    },
    MOVE_BACKWARD {
        @Override
        public void execute(MinecraftClient client) {
            if (client.options != null) client.options.backKey.setPressed(true);
        }
    },
    LOOK_LEFT {
        @Override
        public void execute(MinecraftClient client) {
            if (client.player != null) client.player.setYaw(client.player.getYaw() - 6.0f);
        }
    },
    LOOK_RIGHT {
        @Override
        public void execute(MinecraftClient client) {
            if (client.player != null) client.player.setYaw(client.player.getYaw() + 6.0f);
        }
    },
    LOOK_UP {
        @Override
        public void execute(MinecraftClient client) {
            if (client.player != null) client.player.setPitch(Math.max(-90f, client.player.getPitch() - 6.0f));
        }
    },
    LOOK_DOWN {
        @Override
        public void execute(MinecraftClient client) {
            if (client.player != null) client.player.setPitch(Math.min(90f, client.player.getPitch() + 6.0f));
        }
    },
    IDLE {
        @Override
        public void execute(MinecraftClient client) {
            // Take no action this tick
        }
    };

    public abstract void execute(MinecraftClient client);
}
