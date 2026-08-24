package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.Hand;

public class ActionExecutor {
    private static int attackCooldown = 0;
    private static int sprintTicks = 0;

    public static void execute(int actionIndex, MinecraftClient client) {
        if (client.player == null) return;
        ClientPlayerEntity p = client.player;

        if (attackCooldown > 0) attackCooldown--;
        if (sprintTicks > 0) {
            p.setSprinting(true);
            sprintTicks--;
        }

        // Action Mapping
        // 0: Do Nothing
        // 1: Attack
        // 2: Sprint
        // 3: Jump
        // 4: Aim at opponent

        switch(actionIndex) {
            case 0:
                break;
            case 1:
                doAttack(client);
                break;
            case 2:
                sprintTicks = 10;
                p.setSprinting(true);
                break;
            case 3:
                if (p.isOnGround()) {
                    p.jump();
                }
                break;
            case 4:
                aimAtOpponent(client);
                break;
        }
    }

    private static void doAttack(MinecraftClient client) {
        if (attackCooldown == 0 && client.interactionManager != null && client.player != null) {
            PlayerEntity opp = getClosestOpponent(client);
            if (opp != null && client.player.squaredDistanceTo(opp) < 36.0) {
                client.interactionManager.attackEntity(client.player, opp);
                client.player.swingHand(Hand.MAIN_HAND);
                attackCooldown = 12; // Basic cooldown to prevent spam
            }
        }
    }

    private static void aimAtOpponent(MinecraftClient client) {
        ClientPlayerEntity self = client.player;
        PlayerEntity opponent = getClosestOpponent(client);
        if (self == null || opponent == null) return;

        Vec3d toTarget = new Vec3d(opponent.getX(), opponent.getY(), opponent.getZ())
                .add(0, opponent.getStandingEyeHeight() * 0.5, 0)
                .subtract(self.getEyePos())
                .normalize();

        double yaw = Math.toDegrees(Math.atan2(-toTarget.x, toTarget.z));
        double pitch = Math.toDegrees(-Math.asin(toTarget.y));

        self.setYaw((float) yaw);
        self.setPitch((float) pitch);
    }

    private static PlayerEntity getClosestOpponent(MinecraftClient client) {
        if (client.world == null || client.player == null) return null;
        PlayerEntity closest = null;
        double minDist = Double.MAX_VALUE;

        for (PlayerEntity p : client.world.getPlayers()) {
            if (p == client.player || p.isSpectator()) continue;
            double d = p.squaredDistanceTo(client.player);
            if (d < minDist) {
                minDist = d;
                closest = p;
            }
        }
        return closest;
    }
}
