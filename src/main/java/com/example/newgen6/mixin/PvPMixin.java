package com.example.newgen6.mixin;

import com.example.newgen6.rl.env.PvPStateStore;
import com.example.newgen6.rl.env.PvPSnapshot;
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
    private void newgen6$capturePvPState(CallbackInfo ci) {

        ClientPlayerEntity self =
                (ClientPlayerEntity) (Object) this;

        MinecraftClient client =
                MinecraftClient.getInstance();

        ClientWorld world = client.world;

        if (world == null || !self.isAlive()) {
            PvPStateStore.setLatestSnapshot(
                    PvPSnapshot.empty()
            );
            return;
        }

        OtherClientPlayerEntity nearestTarget = null;

        double nearestDistanceSq =
                Double.POSITIVE_INFINITY;

        for (PlayerEntity candidate : world.getPlayers()) {

            if (candidate == self) {
                continue;
            }

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

        Vec3d selfVelocity =
                self.getVelocity();

        PvPSnapshot snapshot;

        if (nearestTarget == null) {

            snapshot = new PvPSnapshot(
                    true,

                    self.getX(),
                    self.getY(),
                    self.getZ(),

                    selfVelocity.x,
                    selfVelocity.y,
                    selfVelocity.z,

                    self.getYaw(),
                    self.getPitch(),

                    self.getHealth(),
                    self.getAbsorptionAmount(),

                    self.isOnGround(),
                    self.isSprinting(),
                    self.isSneaking(),

                    self.getAttackCooldownProgress(0.0f),

                    false,

                    0.0,
                    0.0,
                    0.0,

                    0.0,
                    0.0,
                    0.0,

                    0.0f,
                    0.0f,

                    0.0f,
                    0.0f,

                    false,
                    false,

                    0.0,

                    self.getYaw()
            );

        } else {

            Vec3d targetVelocity =
                    nearestTarget.getVelocity();

            double relativeX =
                    nearestTarget.getX() - self.getX();

            double relativeY =
                    nearestTarget.getY() - self.getY();

            double relativeZ =
                    nearestTarget.getZ() - self.getZ();

            snapshot = new PvPSnapshot(
                    true,

                    self.getX(),
                    self.getY(),
                    self.getZ(),

                    selfVelocity.x,
                    selfVelocity.y,
                    selfVelocity.z,

                    self.getYaw(),
                    self.getPitch(),

                    self.getHealth(),
                    self.getAbsorptionAmount(),

                    self.isOnGround(),
                    self.isSprinting(),
                    self.isSneaking(),

                    self.getAttackCooldownProgress(0.0f),

                    true,

                    relativeX,
                    relativeY,
                    relativeZ,

                    targetVelocity.x,
                    targetVelocity.y,
                    targetVelocity.z,

                    nearestTarget.getHealth(),
                    nearestTarget.getAbsorptionAmount(),

                    nearestTarget.getYaw(),
                    nearestTarget.getPitch(),

                    nearestTarget.isOnGround(),
                    nearestTarget.isSprinting(),

                    Math.sqrt(nearestDistanceSq),

                    self.getYaw()
            );
        }

        PvPStateStore.setLatestSnapshot(snapshot);
    }
}