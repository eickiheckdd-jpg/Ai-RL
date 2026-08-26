package com.example.newgen6.mixin;

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

    private static volatile Snapshot latestSnapshot = Snapshot.empty();

    @Inject(
        method = "tick",
        at = @At("TAIL")
    )
    private void newgen6$capturePvPState(CallbackInfo ci) {
        ClientPlayerEntity self = (ClientPlayerEntity) (Object) this;

        // In Minecraft 1.21.11 Yarn, use the client world directly rather
        // than relying on a getWorld() method that is not present here.
        MinecraftClient client = MinecraftClient.getInstance();
        ClientWorld world = client.world;

        if (world == null || !self.isAlive()) {
            latestSnapshot = Snapshot.empty();
            return;
        }

        OtherClientPlayerEntity nearestTarget = null;
        double nearestDistanceSq = Double.POSITIVE_INFINITY;

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

            double distanceSq = self.squaredDistanceTo(other);

            if (distanceSq < nearestDistanceSq) {
                nearestDistanceSq = distanceSq;
                nearestTarget = other;
            }
        }

        Vec3d selfVelocity = self.getVelocity();

        Snapshot snapshot;

        if (nearestTarget == null) {
            snapshot = new Snapshot(
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

                0.0f
            );
        } else {
            Vec3d targetVelocity = nearestTarget.getVelocity();

            double relativeX = nearestTarget.getX() - self.getX();
            double relativeY = nearestTarget.getY() - self.getY();
            double relativeZ = nearestTarget.getZ() - self.getZ();

            snapshot = new Snapshot(
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

        latestSnapshot = snapshot;
    }

    public static Snapshot getLatestSnapshot() {
        return latestSnapshot;
    }

    public static final class Snapshot {

        public final boolean valid;

        public final double selfX;
        public final double selfY;
        public final double selfZ;

        public final double selfVelocityX;
        public final double selfVelocityY;
        public final double selfVelocityZ;

        public final float selfYaw;
        public final float selfPitch;

        public final float selfHealth;
        public final float selfAbsorption;
        public final boolean selfOnGround;
        public final boolean selfSprinting;
        public final boolean selfSneaking;
        public final float selfAttackCooldown;

        public final boolean targetPresent;

        public final double targetRelativeX;
        public final double targetRelativeY;
        public final double targetRelativeZ;

        public final double targetVelocityX;
        public final double targetVelocityY;
        public final double targetVelocityZ;

        public final float targetHealth;
        public final float targetAbsorption;
        public final float targetYaw;
        public final float targetPitch;
        public final boolean targetOnGround;
        public final boolean targetSprinting;

        public final double targetDistance;
        public final float selfYawForTargetContext;

        public Snapshot(
            boolean valid,

            double selfX,
            double selfY,
            double selfZ,

            double selfVelocityX,
            double selfVelocityY,
            double selfVelocityZ,

            float selfYaw,
            float selfPitch,

            float selfHealth,
            float selfAbsorption,

            boolean selfOnGround,
            boolean selfSprinting,
            boolean selfSneaking,

            float selfAttackCooldown,

            boolean targetPresent,

            double targetRelativeX,
            double targetRelativeY,
            double targetRelativeZ,

            double targetVelocityX,
            double targetVelocityY,
            double targetVelocityZ,

            float targetHealth,
            float targetAbsorption,

            float targetYaw,
            float targetPitch,

            boolean targetOnGround,
            boolean targetSprinting,

            double targetDistance,
            float selfYawForTargetContext
        ) {
            this.valid = valid;

            this.selfX = selfX;
            this.selfY = selfY;
            this.selfZ = selfZ;

            this.selfVelocityX = selfVelocityX;
            this.selfVelocityY = selfVelocityY;
            this.selfVelocityZ = selfVelocityZ;

            this.selfYaw = selfYaw;
            this.selfPitch = selfPitch;

            this.selfHealth = selfHealth;
            this.selfAbsorption = selfAbsorption;

            this.selfOnGround = selfOnGround;
            this.selfSprinting = selfSprinting;
            this.selfSneaking = selfSneaking;

            this.selfAttackCooldown = selfAttackCooldown;

            this.targetPresent = targetPresent;

            this.targetRelativeX = targetRelativeX;
            this.targetRelativeY = targetRelativeY;
            this.targetRelativeZ = targetRelativeZ;

            this.targetVelocityX = targetVelocityX;
            this.targetVelocityY = targetVelocityY;
            this.targetVelocityZ = targetVelocityZ;

            this.targetHealth = targetHealth;
            this.targetAbsorption = targetAbsorption;

            this.targetYaw = targetYaw;
            this.targetPitch = targetPitch;

            this.targetOnGround = targetOnGround;
            this.targetSprinting = targetSprinting;

            this.targetDistance = targetDistance;
            this.selfYawForTargetContext = selfYawForTargetContext;
        }

        public static Snapshot empty() {
            return new Snapshot(
                false,

                0.0, 0.0, 0.0,

                0.0, 0.0, 0.0,

                0.0f, 0.0f,

                0.0f, 0.0f,

                false, false, false,

                0.0f,

                false,

                0.0, 0.0, 0.0,

                0.0, 0.0, 0.0,

                0.0f, 0.0f,

                0.0f, 0.0f,

                false, false,

                0.0,

                0.0f
            );
        }
    }
}