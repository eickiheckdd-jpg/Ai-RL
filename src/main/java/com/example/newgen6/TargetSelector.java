package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import java.util.List;

public class TargetSelector {
    private LivingEntity currentTarget = null;
    private static final double MAX_TARGET_DISTANCE = 16.0;

    public LivingEntity findBestTarget(MinecraftClient client) {
        if (client.player == null || client.world == null) return null;

        if (currentTarget != null && currentTarget.isAlive() 
            && client.player.squaredDistanceTo(currentTarget) <= MAX_TARGET_DISTANCE * MAX_TARGET_DISTANCE) {
            return currentTarget;
        }

        LivingEntity bestTarget = null;
        double closestDistanceSq = MAX_TARGET_DISTANCE * MAX_TARGET_DISTANCE;

        List<Entity> entities = client.world.getOtherEntities(client.player, client.player.getBoundingBox().expand(MAX_TARGET_DISTANCE));
        for (Entity entity : entities) {
            if (entity instanceof LivingEntity living && living.isAlive() && entity != client.player) {
                if (living instanceof PlayerEntity || living.isHostile()) {
                    double distSq = client.player.squaredDistanceTo(living);
                    if (distSq < closestDistanceSq) {
                        closestDistanceSq = distSq;
                        bestTarget = living;
                    }
                }
            }
        }
        this.currentTarget = bestTarget;
        return bestTarget;
    }

    public LivingEntity getCurrentTarget() {
        return currentTarget;
    }
}
