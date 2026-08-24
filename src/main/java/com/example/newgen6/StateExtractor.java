package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Vec3d;

public class StateExtractor {
    public static double[] extractState(MinecraftClient client) {
        double[] s = new double[11];
        ClientPlayerEntity self = client.player;
        if (self == null) return s;

        s[0] = self.getHealth();
        s[1] = self.getArmor();
        s[2] = self.fallDistance;
        s[3] = self.isOnGround() ? 1.0 : 0.0;
        s[4] = self.isSprinting() ? 1.0 : 0.0;

        PlayerEntity opponent = getClosestOpponent(client);
        if (opponent != null) {
            Vec3d selfPos = new Vec3d(self.getX(), self.getY(), self.getZ());
            Vec3d opponentPos = new Vec3d(opponent.getX(), opponent.getY(), opponent.getZ());
            
            Vec3d diff = opponentPos.subtract(selfPos);
            s[5] = diff.x;
            s[6] = diff.y;
            s[7] = diff.z;
            s[8] = diff.length();
            s[9] = opponent.getHealth();
            s[10] = opponent.isBlocking() ? 1.0 : 0.0;
        }

        return s;
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
