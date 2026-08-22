package com.example.newgen6.client;

import com.example.newgen6.rl.BotAction;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;

/**
 * Translates a BotAction into real input on the local player — the same
 * effect as if the player pressed the corresponding keys/clicked. This
 * does NOT spoof packets directly; it drives the same input/interaction
 * paths vanilla input handling uses, so server-side anti-cheat sees normal
 * player movement.
 *
 * NOTE: player.input field names (movementForward/movementSideways/jumping)
 * and InteractionManager#attackEntity are Yarn conventions that may have
 * shifted slightly by 1.21.11 — verify against your decompiled sources.
 */
public class PlayerActionHandler {

    private static final float TURN_SPEED_DEG = 8.0f;

    public static void apply(MinecraftClient client, BotAction action, Entity target) {
        ClientPlayerEntity player = client.player;
        if (player == null) return;

        // Reset per-tick movement input before applying the new action —
        // otherwise input state would stick from the previous tick.
        player.input.movementForward = 0.0f;
        player.input.movementSideways = 0.0f;
        player.input.jumping = false;

        switch (action) {
            case MOVE_FORWARD -> player.input.movementForward = 1.0f;
            case MOVE_BACK -> player.input.movementForward = -1.0f;
            case STRAFE_LEFT -> player.input.movementSideways = 1.0f;
            case STRAFE_RIGHT -> player.input.movementSideways = -1.0f;
            case JUMP -> player.input.jumping = true;
            case ATTACK -> attack(client, player, target);
            case BLOCK -> { /* raise shield — wire to your item-use logic if using a shield item */ }
            case LOOK_LEFT -> player.setYaw(player.getYaw() - TURN_SPEED_DEG);
            case LOOK_RIGHT -> player.setYaw(player.getYaw() + TURN_SPEED_DEG);
            case NO_OP -> { /* nothing */ }
        }
    }

    /** Optional: smoothly turn toward a target rather than snapping — call instead of raw LOOK actions if desired. */
    public static void smoothAimAt(ClientPlayerEntity player, Entity target, double maxDegreesPerTick) {
        double dx = target.getX() - player.getX();
        double dz = target.getZ() - player.getZ();
        double dy = (target.getY() + target.getStandingEyeHeight())
                - (player.getY() + player.getStandingEyeHeight());
        double horizontalDist = Math.sqrt(dx * dx + dz * dz);

        float desiredYaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0);
        float desiredPitch = (float) -Math.toDegrees(Math.atan2(dy, horizontalDist));

        player.setYaw(stepTowards(player.getYaw(), desiredYaw, (float) maxDegreesPerTick));
        player.setPitch(stepTowards(player.getPitch(), desiredPitch, (float) maxDegreesPerTick));
    }

    private static float stepTowards(float current, float target, float maxStep) {
        float delta = wrapDegrees(target - current);
        delta = Math.max(-maxStep, Math.min(maxStep, delta));
        return current + delta;
    }

    private static float wrapDegrees(float degrees) {
        degrees %= 360.0f;
        if (degrees >= 180.0f) degrees -= 360.0f;
        if (degrees < -180.0f) degrees += 360.0f;
        return degrees;
    }

    private static void attack(MinecraftClient client, ClientPlayerEntity player, Entity target) {
        if (target == null || client.interactionManager == null) return;
        client.interactionManager.attackEntity(player, target);
        player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
    }
}
