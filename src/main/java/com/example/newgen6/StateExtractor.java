package com.example.newgen6.game;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.util.List;

public class StateExtractor {

    public static final int STATE_SIZE = 11;
    public static final double SEARCH_RADIUS = 16.0;
    public static final double MAX_HEALTH = 20.0;

    /** Nearest other player within SEARCH_RADIUS, or null if none is visible/nearby. */
    public PlayerEntity findOpponent(MinecraftClient client) {
        if (client.player == null || client.world == null) return null;
        Box box = client.player.getBoundingBox().expand(SEARCH_RADIUS);
        // NOTE: getEntitiesByClass signature has stayed stable across recent
        // versions, but double check against your exact mappings if this
        // doesn't compile.
        List<PlayerEntity> nearby = client.world.getEntitiesByClass(
                PlayerEntity.class, box, e -> e != client.player && e.isAlive());

        PlayerEntity closest = null;
        double closestDistSq = Double.MAX_VALUE;
        for (PlayerEntity p : nearby) {
            double d = p.squaredDistanceTo(client.player);
            if (d < closestDistSq) {
                closestDistSq = d;
                closest = p;
            }
        }
        return closest;
    }

    public double[] extract(MinecraftClient client, PlayerEntity opponent) {
        double[] s = new double[STATE_SIZE];
        ClientPlayerEntity self = client.player;
        if (self == null || opponent == null) return s;

        Vec3d diff = opponent.getPos().subtract(self.getPos());
        double distance = diff.length();

        s[0] = clamp(diff.x / SEARCH_RADIUS);
        s[1] = clamp(diff.y / SEARCH_RADIUS);
        s[2] = clamp(diff.z / SEARCH_RADIUS);
        s[3] = clamp(distance / SEARCH_RADIUS);
        s[4] = normAngle(self.getYaw());
        s[5] = normAngle(self.getPitch());
        s[6] = self.getHealth() / MAX_HEALTH;
        s[7] = opponent.getHealth() / MAX_HEALTH;
        s[8] = self.isOnGround() ? 1.0 : 0.0;
        // NOTE: canSee(Entity) lives on LivingEntity in recent mappings.
        s[9] = self.canSee(opponent) ? 1.0 : 0.0;
        // NOTE: getAttackCooldownProgress(float) is the standard PlayerEntity method.
        s[10] = self.getAttackCooldownProgress(0.0f);
        return s;
    }

    private double clamp(double v) {
        return Math.max(-1.0, Math.min(1.0, v));
    }

    private double normAngle(float deg) {
        double d = ((deg + 180) % 360 + 360) % 360 - 180;
        return d / 180.0;
    }
}
