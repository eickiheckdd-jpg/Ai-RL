package com.example.newgen6.rl.env;

public final class PvPSnapshot {

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

    public PvPSnapshot(
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

        this.selfAttackCooldown =
                selfAttackCooldown;

        this.targetPresent =
                targetPresent;

        this.targetRelativeX =
                targetRelativeX;

        this.targetRelativeY =
                targetRelativeY;

        this.targetRelativeZ =
                targetRelativeZ;

        this.targetVelocityX =
                targetVelocityX;

        this.targetVelocityY =
                targetVelocityY;

        this.targetVelocityZ =
                targetVelocityZ;

        this.targetHealth =
                targetHealth;

        this.targetAbsorption =
                targetAbsorption;

        this.targetYaw =
                targetYaw;

        this.targetPitch =
                targetPitch;

        this.targetOnGround =
                targetOnGround;

        this.targetSprinting =
                targetSprinting;

        this.targetDistance =
                targetDistance;

        this.selfYawForTargetContext =
                selfYawForTargetContext;
    }

    public static PvPSnapshot empty() {
        return new PvPSnapshot(
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
                false,

                0.0f,

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
    }
}