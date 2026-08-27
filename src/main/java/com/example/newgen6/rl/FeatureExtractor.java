package com.example.newgen6.rl;

/**
 * Builds the exact 229-dimensional observation vector for HT3 Sword.
 * All features are cheap (no expensive raycasts every tick).
 *
 * Layout (0-based indices):
 *  0-34   Self state (35)
 * 35-89   Relative target (55)
 * 90-149  Combat history ring buffer (60)
 * 150-179 Movement / spacing (30)
 * 180-204 Simple environment (25)
 * 205-224 Timing / rhythm (20)
 * 225-228 Derived / misc (4)
 */
public final class FeatureExtractor {

    public static final int DIM = Config.OBS_DIM;

    // Ring buffers for history (length 10)
    private final float[] histHit = new float[10];
    private final float[] histCrit = new float[10];
    private final float[] histKbX = new float[10];
    private final float[] histKbZ = new float[10];
    private final float[] histDmgDealt = new float[10];
    private final float[] histDmgTaken = new float[10];
    private int histIdx = 0;

    private float lastYaw, lastPitch;
    private long lastAttackTick, lastHurtTick, lastJumpTick, lastCritTick;
    private float comboLength;
    private float prevDist = 3.0f;

    public void reset() {
        for (int i = 0; i < 10; i++) {
            histHit[i] = histCrit[i] = histKbX[i] = histKbZ[i] = 0f;
            histDmgDealt[i] = histDmgTaken[i] = 0f;
        }
        histIdx = 0;
        comboLength = 0f;
        lastAttackTick = lastHurtTick = lastJumpTick = lastCritTick = 0;
        prevDist = 3.0f;
    }

    /**
     * Fill obs[0..228]. Caller must supply current game state via the Context.
     */
    public void extract(ObsContext ctx, float[] obs) {
        int i = 0;

        // ========== SELF (0-34) ==========
        obs[i++] = ctx.health / 20f;
        obs[i++] = ctx.hunger / 20f;
        obs[i++] = clamp(ctx.velX, -1.5f, 1.5f);
        obs[i++] = clamp(ctx.velY, -1.5f, 1.5f);
        obs[i++] = clamp(ctx.velZ, -1.5f, 1.5f);
        obs[i++] = (float) Math.sin(Math.toRadians(ctx.yaw));
        obs[i++] = (float) Math.cos(Math.toRadians(ctx.yaw));
        obs[i++] = (float) Math.sin(Math.toRadians(ctx.pitch));
        obs[i++] = (float) Math.cos(Math.toRadians(ctx.pitch));
        obs[i++] = ctx.attackCooldown;               // 0..1
        obs[i++] = ctx.onGround ? 1f : 0f;
        obs[i++] = ctx.sprinting ? 1f : 0f;
        obs[i++] = ctx.sneaking ? 1f : 0f;
        obs[i++] = ctx.jumping ? 1f : 0f;
        obs[i++] = clamp(ctx.fallDistance / 10f, 0f, 1f);
        obs[i++] = ticksSince(ctx.tick, lastHurtTick) / 40f;
        obs[i++] = ticksSince(ctx.tick, lastAttackTick) / 20f;
        obs[i++] = ticksSince(ctx.tick, lastJumpTick) / 20f;
        obs[i++] = ticksSince(ctx.tick, lastCritTick) / 30f;
        obs[i++] = comboLength / 8f;
        // velocity magnitude + horizontal
        float speed = (float) Math.sqrt(ctx.velX * ctx.velX + ctx.velZ * ctx.velZ);
        obs[i++] = clamp(speed / 0.3f, 0f, 2f);
        obs[i++] = clamp(ctx.velY / 0.5f, -2f, 2f);
        // look delta (how fast we are turning)
        float dyaw = wrapDegrees(ctx.yaw - lastYaw);
        float dpitch = ctx.pitch - lastPitch;
        obs[i++] = clamp(dyaw / 30f, -1f, 1f);
        obs[i++] = clamp(dpitch / 20f, -1f, 1f);
        lastYaw = ctx.yaw;
        lastPitch = ctx.pitch;
        // remaining self padding / extras
        obs[i++] = ctx.isInWater ? 1f : 0f;
        obs[i++] = ctx.isOnFire ? 1f : 0f;
        obs[i++] = ctx.absorption / 8f;
        obs[i++] = ctx.hasSpeedEffect ? 1f : 0f;
        obs[i++] = ctx.hasSlowness ? 1f : 0f;
        obs[i++] = ctx.hasJumpBoost ? 1f : 0f;
        obs[i++] = 0f; // reserved
        obs[i++] = 0f;
        obs[i++] = 0f;
        obs[i++] = 0f;
        obs[i++] = 0f; // 35 features (0-34)

        // ========== RELATIVE TARGET (35-89) ==========
        if (ctx.hasTarget) {
            float dx = ctx.tDx, dy = ctx.tDy, dz = ctx.tDz;
            float dist = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
            float hDist = (float) Math.sqrt(dx * dx + dz * dz);
            obs[i++] = clamp(dx / 6f, -1f, 1f);
            obs[i++] = clamp(dy / 4f, -1f, 1f);
            obs[i++] = clamp(dz / 6f, -1f, 1f);
            obs[i++] = clamp(dist / 6f, 0f, 1.5f);
            obs[i++] = clamp(hDist / 6f, 0f, 1.5f);
            obs[i++] = clamp(ctx.tVelX, -1.5f, 1.5f);
            obs[i++] = clamp(ctx.tVelY, -1.5f, 1.5f);
            obs[i++] = clamp(ctx.tVelZ, -1.5f, 1.5f);
            // relative velocity
            obs[i++] = clamp(ctx.tVelX - ctx.velX, -2f, 2f);
            obs[i++] = clamp(ctx.tVelZ - ctx.velZ, -2f, 2f);
            // angle to target
            float angleTo = (float) Math.toDegrees(Math.atan2(dz, dx)) - ctx.yaw;
            angleTo = wrapDegrees(angleTo);
            obs[i++] = (float) Math.sin(Math.toRadians(angleTo));
            obs[i++] = (float) Math.cos(Math.toRadians(angleTo));
            obs[i++] = clamp(angleTo / 90f, -1f, 1f);
            // pitch needed
            float pitchTo = (float) Math.toDegrees(Math.atan2(dy, hDist));
            obs[i++] = clamp((pitchTo - ctx.pitch) / 45f, -1f, 1f);
            obs[i++] = ctx.isLookingAtTarget ? 1f : 0f;
            obs[i++] = ctx.targetHealth / 20f;
            obs[i++] = ctx.targetHurtTime / 10f;
            obs[i++] = ctx.targetAttackCooldown;
            obs[i++] = ctx.targetSprinting ? 1f : 0f;
            obs[i++] = ctx.targetOnGround ? 1f : 0f;
            obs[i++] = ctx.targetSneaking ? 1f : 0f;
            // optimal spacing signal (Sword ideal ~2.9-3.3)
            float spacingErr = hDist - 3.1f;
            obs[i++] = clamp(spacingErr / 2f, -1.5f, 1.5f);
            obs[i++] = (hDist > 2.6f && hDist < 3.5f) ? 1f : 0f;
            // closing / opening speed
            float closing = prevDist - hDist;
            obs[i++] = clamp(closing * 10f, -1f, 1f);
            prevDist = hDist;
            // more relative
            for (int k = 0; k < 30; k++) obs[i++] = 0f; // reserved / future
        } else {
            for (int k = 0; k < 55; k++) obs[i++] = 0f;
        }

        // ========== COMBAT HISTORY (90-149) ==========
        for (int h = 0; h < 10; h++) {
            int idx = (histIdx - 1 - h + 10) % 10;
            obs[i++] = histHit[idx];
            obs[i++] = histCrit[idx];
            obs[i++] = histKbX[idx];
            obs[i++] = histKbZ[idx];
            obs[i++] = histDmgDealt[idx];
            obs[i++] = histDmgTaken[idx];
        }

        // ========== MOVEMENT / SPACING (150-179) ==========
        obs[i++] = ctx.forwardPressed ? 1f : 0f;
        obs[i++] = ctx.backPressed ? 1f : 0f;
        obs[i++] = ctx.leftPressed ? 1f : 0f;
        obs[i++] = ctx.rightPressed ? 1f : 0f;
        obs[i++] = ctx.jumpPressed ? 1f : 0f;
        // predicted positions etc. simple
        for (int k = 0; k < 25; k++) obs[i++] = 0f;

        // ========== ENVIRONMENT (180-204) ==========
        obs[i++] = ctx.blockUnderFeet;
        obs[i++] = ctx.blockInFront;
        obs[i++] = ctx.blockAbove;
        obs[i++] = ctx.nearEdge ? 1f : 0f;
        obs[i++] = ctx.inLiquid ? 1f : 0f;
        for (int k = 0; k < 20; k++) obs[i++] = 0f;

        // ========== TIMING (205-224) ==========
        obs[i++] = ctx.attackCooldown;
        obs[i++] = (ctx.attackCooldown > 0.9f) ? 1f : 0f; // ready to hit
        obs[i++] = ticksSince(ctx.tick, lastCritTick) / 40f;
        obs[i++] = ticksSince(ctx.tick, lastJumpTick) / 15f;
        for (int k = 0; k < 16; k++) obs[i++] = 0f;

        // ========== DERIVED (225-228) ==========
        obs[i++] = ctx.hasTarget ? 1f : 0f;
        obs[i++] = (comboLength > 2f) ? 1f : 0f;
        obs[i++] = 0f;
        obs[i++] = 0f;

        // safety
        while (i < DIM) obs[i++] = 0f;
    }

    public void recordHit(boolean crit, float kbX, float kbZ, float dmgDealt) {
        histHit[histIdx] = 1f;
        histCrit[histIdx] = crit ? 1f : 0f;
        histKbX[histIdx] = clamp(kbX, -1f, 1f);
        histKbZ[histIdx] = clamp(kbZ, -1f, 1f);
        histDmgDealt[histIdx] = clamp(dmgDealt / 10f, 0f, 1f);
        histIdx = (histIdx + 1) % 10;
        if (crit) lastCritTick = System.currentTimeMillis() / 50; // rough
        comboLength = Math.min(comboLength + 1f, 12f);
    }

    public void recordHurt(float dmg) {
        histDmgTaken[histIdx] = clamp(dmg / 10f, 0f, 1f);
        comboLength = 0f;
    }

    public void onAttack(long tick) { lastAttackTick = tick; }
    public void onJump(long tick) { lastJumpTick = tick; }
    public void onHurt(long tick) { lastHurtTick = tick; }
    public void onCrit(long tick) { lastCritTick = tick; }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    private static float wrapDegrees(float deg) {
        deg %= 360f;
        if (deg >= 180f) deg -= 360f;
        if (deg < -180f) deg += 360f;
        return deg;
    }

    private static float ticksSince(long now, long then) {
        if (then == 0) return 100f;
        return (float) Math.min(100, now - then);
    }

    /** Simple data carrier filled by the Mixin / client each tick */
    public static final class ObsContext {
        public long tick;
        public float health, hunger, absorption;
        public float velX, velY, velZ;
        public float yaw, pitch;
        public float attackCooldown;
        public boolean onGround, sprinting, sneaking, jumping;
        public float fallDistance;
        public boolean isInWater, isOnFire, hasSpeedEffect, hasSlowness, hasJumpBoost;

        public boolean hasTarget;
        public float tDx, tDy, tDz;
        public float tVelX, tVelY, tVelZ;
        public float targetHealth, targetHurtTime, targetAttackCooldown;
        public boolean isLookingAtTarget, targetSprinting, targetOnGround, targetSneaking;

        public boolean forwardPressed, backPressed, leftPressed, rightPressed, jumpPressed;
        public float blockUnderFeet, blockInFront, blockAbove;
        public boolean nearEdge, inLiquid;
    }
}
