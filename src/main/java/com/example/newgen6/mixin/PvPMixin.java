package com.example.newgen6.mixin;

import com.example.newgen6.rl.env.PvPSnapshot;
import com.example.newgen6.rl.env.PvPStateStore;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.OtherClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Vec3d;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientPlayerEntity.class)
public abstract class PvPMixin {

    @Inject(
            method = "tick",
            at = @At("TAIL")
    )
    private void newgen6$capturePvPState(
            CallbackInfo ci) {

        ClientPlayerEntity self =
                (ClientPlayerEntity) (Object) this;

        MinecraftClient client =
                MinecraftClient.getInstance();

        if (client == null) {
            PvPStateStore.setLatestSnapshot(
                    PvPSnapshot.empty()
            );
            return;
        }

        ClientWorld world = client.world;

        /*
         * No usable Minecraft state.
         */
        if (world == null || !self.isAlive()) {
            PvPStateStore.setLatestSnapshot(
                    PvPSnapshot.empty()
            );
            return;
        }

        /*
         * Find the nearest valid player target.
         */
        OtherClientPlayerEntity nearestTarget = null;

        double nearestDistanceSq =
                Double.POSITIVE_INFINITY;

        for (PlayerEntity candidate : world.getPlayers()) {

            if (candidate == self) {
                continue;
            }

            /*
             * Only capture actual other client players.
             */
            if (!(candidate instanceof OtherClientPlayerEntity other)) {
                continue;
            }

            if (!other.isAlive() || other.isSpectator()) {
                continue;
            }

            double distanceSq =
                    self.squaredDistanceTo(other);

            if (distanceSq < nearestDistanceSq) {
                nearestDistanceSq = distanceSq;
                nearestTarget = other;
            }
        }

        /*
         * Self state.
         */
        Vec3d selfVelocity =
                self.getVelocity();

        PvPSnapshot snapshot;

        /*
         * --------------------------------------------------
         * No target
         * --------------------------------------------------
         */
        if (nearestTarget == null) {

            snapshot = new PvPSnapshot(
                    true,

                    // Self position
                    self.getX(),
                    self.getY(),
                    self.getZ(),

                    // Self velocity
                    selfVelocity.x,
                    selfVelocity.y,
                    selfVelocity.z,

                    // Self rotation
                    self.getYaw(),
                    self.getPitch(),

                    // Self health
                    self.getHealth(),
                    self.getAbsorptionAmount(),

                    // Self movement state
                    self.isOnGround(),
                    self.isSprinting(),
                    self.isSneaking(),

                    // Attack cooldown
                    self.getAttackCooldownProgress(0.0f),

                    // Target
                    false,

                    // Target relative position
                    0.0,
                    0.0,
                    0.0,

                    // Target velocity
                    0.0,
                    0.0,
                    0.0,

                    // Target health
                    0.0f,
                    0.0f,

                    // Target rotation
                    0.0f,
                    0.0f,

                    // Target movement
                    false,
                    false,

                    // Target distance
                    0.0
            );

        }

        /*
         * --------------------------------------------------
         * Target found
         * --------------------------------------------------
         */
        else {

            Vec3d targetVelocity =
                    nearestTarget.getVelocity();

            /*
             * Target position relative to the player.
             */
            double relativeX =
                    nearestTarget.getX()
                            - self.getX();

            double relativeY =
                    nearestTarget.getY()
                            - self.getY();

            double relativeZ =
                    nearestTarget.getZ()
                            - self.getZ();

            snapshot = new PvPSnapshot(
                    true,

                    // Self position
                    self.getX(),
                    self.getY(),
                    self.getZ(),

                    // Self velocity
                    selfVelocity.x,
                    selfVelocity.y,
                    selfVelocity.z,

                    // Self rotation
                    self.getYaw(),
                    self.getPitch(),

                    // Self health
                    self.getHealth(),
                    self.getAbsorptionAmount(),

                    // Self movement state
                    self.isOnGround(),
                    self.isSprinting(),
                    self.isSneaking(),

                    // Attack cooldown
                    self.getAttackCooldownProgress(0.0f),

                    // Target exists
                    true,

                    // Target relative position
                    relativeX,
                    relativeY,
                    relativeZ,

                    // Target velocity
                    targetVelocity.x,
                    targetVelocity.y,
                    targetVelocity.z,

                    // Target health
                    nearestTarget.getHealth(),
                    nearestTarget.getAbsorptionAmount(),

                    // Target rotation
                    nearestTarget.getYaw(),
                    nearestTarget.getPitch(),

                    // Target movement
                    nearestTarget.isOnGround(),
                    nearestTarget.isSprinting(),

                    // Target distance
                    Math.sqrt(nearestDistanceSq)
            );
        }

        /*
         * Publish the immutable snapshot.
         *
         * PvPMixin does not perform RL decisions.
         */
        PvPStateStore.setLatestSnapshot(snapshot);
    }
}