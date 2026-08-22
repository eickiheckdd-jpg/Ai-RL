package com.example.newgen6.client;

import com.example.newgen6.rl.BotAction;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.util.PlayerInput;

/**
 * Translates a BotAction into real input on the local player — the same
 * effect as if the player pressed the corresponding keys/clicked. This
 * does NOT spoof packets directly; it drives the same input path vanilla
 * keyboard input feeds into, so server-side anti-cheat sees normal player
 * movement.
 *
 * NOTE: as of the 1.21.9 input rework, Input no longer exposes
 * movementForward/movementSideways/jumping directly — instead you assign a
 * whole PlayerInput record (forward/backward/left/right/jump/sneak/sprint)
 * to player.input.playerInput each tick. This replaces the previous
 * (pre-1.21.9) field-based approach.
 */
public class PlayerActionHandler {

    private static final float TURN_SPEED_DEG = 8.0f;

    public static void apply(MinecraftClient client, BotAction action, Entity target) {
        ClientPlayerEntity player = client.player;
        if (player == null) return;

        boolean forward = false, backward = false, left = false, right = false, jump = false;

        switch (action) {
            case MOVE_FORWARD -> forward = true;
            case MOVE_BACK -> backward = true;
            case STRAFE_LEFT -> left = true;
            case STRAFE_RIGHT -> right = true;
            case JUMP -> jump = true;
            case ATTACK -> attack(client, player, target);
            case BLOCK -> { /* raise shield — wire to your item-use logic if using a shield item */ }
            case LOOK_LEFT -> player.setYaw(player.getYaw() - TURN_SPEED_DEG);
            case LOOK_RIGHT -> player.setYaw(player.getYaw() + TURN_SPEED_DEG);
            case NO_OP -> { /* nothing */ }
        }

        // sneak/sprint left false — extend the action space later if you want the
        // bot to control those independently.
        player.input.playerInput = new PlayerInput(forward, backward, left, right, jump, false, false);
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