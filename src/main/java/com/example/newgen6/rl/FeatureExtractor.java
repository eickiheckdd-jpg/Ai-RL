package com.example.newgen6.rl;

/**
 * Expanded observation for HT3 Sword — richer combat features, still cheap.
 * Dim = Config.OBS_DIM (256).
 *
 * No hardcoded tactics — only raw measurable state the policy can learn from.
 */
public final class FeatureExtractor {

    public static final int DIM = Config.OBS_DIM;

    private final float[] histHit = new float[12];
    private final float[] histCrit = new float[12];
    private final float[] histDmg = new float[12];
    private final float[] histDist = new float[12];
    private final float[] histLookErr = new float[12];
    private int histIdx = 0;

    private float prevTDx, prevTDz, prevHDist = 3.1f;
    private float prevSelfYaw, prevSelfPitch;
    private int ticksSinceHit, ticksSinceHurt, ticksSinceAttack;

    public void reset() {
        for (int i = 0; i < 12; i++) {
            histHit[i] = histCrit[i] = histDmg[i] = 0f;
            histDist[i] = 0.5f;
            histLookErr[i] = 0f;
        }
        histIdx = 0;
        prevTDx = prevTDz = 0f;
        prevHDist = 3.1f;
        ticksSinceHit = ticksSinceHurt = ticksSinceAttack = 99;
    }

    public void extract(ObsContext ctx, float[] obs) {
        int i = 0;

        // ========== SELF (0-39) ==========
        obs[i++] = ctx.health / 20f;
        obs[i++] = ctx.hunger / 20f;
        obs[i++] = ctx.absorption / 16f;
        obs[i++] = clamp(ctx.velX / 0.3f, -1f, 1f);
        obs[i++] = clamp(ctx.velY / 0.5f, -1f, 1f);
        obs[i++] = clamp(ctx.velZ / 0.3f, -1f, 1f);
        float speed = (float) Math.sqrt(ctx.velX * ctx.velX + ctx.velZ * ctx.velZ);
        obs[i++] = clamp(speed / 0.3f, 0f, 1f);
        obs[i++] = ctx.yaw / 180f;
        obs[i++] = ctx.pitch / 90f;
        obs[i++] = ctx.attackCooldown;                    // 0..1 ready
        obs[i++] = 1f - ctx.attackCooldown;                 // time-to-ready proxy
        obs[i++] = ctx.onGround ? 1f : 0f;
        obs[i++] = ctx.sprinting ? 1f : 0f;
        obs[i++] = ctx.sneaking ? 1f : 0f;
        obs[i++] = ctx.jumping ? 1f : 0f;
        obs[i++] = clamp(ctx.fallDistance / 3f, 0f, 1f);
        obs[i++] = (ctx.fallDistance > 0f && !ctx.onGround) ? 1f : 0f; // can-crit window
        obs[i++] = ctx.isInWater ? 1f : 0f;
        obs[i++] = ctx.isOnFire ? 1f : 0f;
        obs[i++] = clamp(ticksSinceHit / 40f, 0f, 1f);
        obs[i++] = clamp(ticksSinceHurt / 40f, 0f, 1f);
        obs[i++] = clamp(ticksSinceAttack / 20f, 0f, 1f);
        // yaw/pitch velocity (how fast we're turning)
        float yawVel = wrapDeg(ctx.yaw - prevSelfYaw);
        float pitchVel = ctx.pitch - prevSelfPitch;
        obs[i++] = clamp(yawVel / 30f, -1f, 1f);
        obs[i++] = clamp(pitchVel / 20f, -1f, 1f);
        prevSelfYaw = ctx.yaw;
        prevSelfPitch = ctx.pitch;
        while (i < 40) obs[i++] = 0f;

        // ========== TARGET RELATIVE (40-99) ==========
        if (ctx.hasTarget) {
            float dx = ctx.tDx, dy = ctx.tDy, dz = ctx.tDz;
            float hDist = (float) Math.sqrt(dx * dx + dz * dz);
            float dist3 = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);

            obs[i++] = 1f; // has target
            obs[i++] = clamp(hDist / 6f, 0f, 1f);
            obs[i++] = clamp(dist3 / 6f, 0f, 1f);
            obs[i++] = clamp(dx / 6f, -1f, 1f);
            obs[i++] = clamp(dy / 3f, -1f, 1f);
            obs[i++] = clamp(dz / 6f, -1f, 1f);
            obs[i++] = clamp(ctx.tVelX / 0.3f, -1f, 1f);
            obs[i++] = clamp(ctx.tVelY / 0.5f, -1f, 1f);
            obs[i++] = clamp(ctx.tVelZ / 0.3f, -1f, 1f);

            // closing speed (positive = approaching)
            float closing = prevHDist - hDist;
            obs[i++] = clamp(closing / 0.2f, -1f, 1f);
            // relative velocity along look direction (approx)
            float yawRad = (float) Math.toRadians(ctx.yaw);
            float lookX = -(float) Math.sin(yawRad);
            float lookZ = (float) Math.cos(yawRad);
            float relVx = ctx.tVelX - ctx.velX;
            float relVz = ctx.tVelZ - ctx.velZ;
            float approach = -(relVx * lookX + relVz * lookZ); // toward us
            obs[i++] = clamp(approach / 0.3f, -1f, 1f);

            // angle to target
            float angleTo = (float) Math.toDegrees(Math.atan2(-(dx), dz));
            float yawErr = wrapDeg(angleTo - ctx.yaw);
            obs[i++] = clamp(yawErr / 90f, -1f, 1f);
            obs[i++] = (float) Math.cos(Math.toRadians(yawErr));
            obs[i++] = (float) Math.sin(Math.toRadians(yawErr));
            float pitchTo = (float) Math.toDegrees(Math.atan2(dy, Math.max(hDist, 0.01f)));
            float pitchErr = pitchTo - ctx.pitch;
            obs[i++] = clamp(pitchErr / 45f, -1f, 1f);
            obs[i++] = ctx.isLookingAtTarget ? 1f : 0f;
            obs[i++] = (Math.abs(yawErr) < 25f && Math.abs(pitchErr) < 30f) ? 1f : 0f;

            // spacing bands (raw info, not rewards)
            obs[i++] = hDist < 2.2f ? 1f : 0f;
            obs[i++] = (hDist >= 2.2f && hDist <= 3.6f) ? 1f : 0f;
            obs[i++] = hDist > 3.6f ? 1f : 0f;
            obs[i++] = clamp((hDist - 3.05f) / 2f, -1f, 1f);

            obs[i++] = ctx.targetHealth / 20f;
            obs[i++] = clamp(ctx.targetHurtTime / 10f, 0f, 1f);
            obs[i++] = ctx.targetAttackCooldown;
            obs[i++] = 1f - ctx.targetAttackCooldown;
            obs[i++] = ctx.targetSprinting ? 1f : 0f;
            obs[i++] = ctx.targetOnGround ? 1f : 0f;
            obs[i++] = ctx.targetSneaking ? 1f : 0f;
            // target airborne / fall (crit threat / opportunity)
            obs[i++] = (!ctx.targetOnGround) ? 1f : 0f;

            // predicted target pos 3 ticks ahead (linear)
            obs[i++] = clamp((dx + ctx.tVelX * 3f) / 6f, -1f, 1f);
            obs[i++] = clamp((dz + ctx.tVelZ * 3f) / 6f, -1f, 1f);

            prevHDist = hDist;
            prevTDx = dx;
            prevTDz = dz;

            // push history
            histDist[histIdx] = clamp(hDist / 6f, 0f, 1f);
            histLookErr[histIdx] = clamp(Math.abs(yawErr) / 90f, 0f, 1f);
        } else {
            for (int k = 0; k < 32; k++) obs[i++] = 0f;
            prevHDist = 3.1f;
        }
        while (i < 100) obs[i++] = 0f;

        // ========== COMBAT HISTORY RING (100-171) — 12 steps × 6 ==========
        for (int h = 0; h < 12; h++) {
            int idx = (histIdx - 1 - h + 12) % 12;
            obs[i++] = histHit[idx];
            obs[i++] = histCrit[idx];
            obs[i++] = histDmg[idx];
            obs[i++] = histDist[idx];
            obs[i++] = histLookErr[idx];
            obs[i++] = 0f; // reserved
        }
        while (i < 172) obs[i++] = 0f;

        // ========== TIMING / RHYTHM (172-199) ==========
        obs[i++] = clamp(ticksSinceHit / 20f, 0f, 1f);
        obs[i++] = clamp(ticksSinceHurt / 20f, 0f, 1f);
        obs[i++] = (ticksSinceHit < 8) ? 1f : 0f;   // recent hit
        obs[i++] = (ticksSinceHurt < 8) ? 1f : 0f;
        obs[i++] = (ctx.attackCooldown >= 0.9f) ? 1f : 0f; // swing ready
        obs[i++] = (ctx.attackCooldown >= 0.9f && ctx.hasTarget && ctx.isLookingAtTarget) ? 1f : 0f;
        // sin/cos of tick for weak phase signal
        float ph = (ctx.tick % 20) / 20f * (float) (Math.PI * 2);
        obs[i++] = (float) Math.sin(ph);
        obs[i++] = (float) Math.cos(ph);
        while (i < 200) obs[i++] = 0f;

        // ========== ENV (200-219) ==========
        obs[i++] = ctx.blockUnderFeet;
        obs[i++] = ctx.inLiquid ? 1f : 0f;
        obs[i++] = ctx.isInWater ? 1f : 0f;
        obs[i++] = ctx.isOnFire ? 1f : 0f;
        while (i < 220) obs[i++] = 0f;

        // ========== MISC / DERIVED (220-255) ==========
        obs[i++] = ctx.hasTarget ? 1f : 0f;
        obs[i++] = clamp(ctx.health / 20f - (ctx.hasTarget ? ctx.targetHealth / 20f : 0.5f), -1f, 1f);
        while (i < DIM) obs[i++] = 0f;

        // advance timers
        ticksSinceHit++;
        ticksSinceHurt++;
        ticksSinceAttack++;
    }

    /** Call when a hit/hurt/attack event is known this tick */
    public void notifyHit(boolean hit, boolean crit, float dmg) {
        histHit[histIdx] = hit ? 1f : 0f;
        histCrit[histIdx] = crit ? 1f : 0f;
        histDmg[histIdx] = clamp(dmg / 10f, 0f, 1f);
        histIdx = (histIdx + 1) % 12;
        if (hit) ticksSinceHit = 0;
    }

    public void notifyHurt() { ticksSinceHurt = 0; }
    public void notifyAttack() { ticksSinceAttack = 0; }

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    private static float wrapDeg(float d) {
        while (d > 180f) d -= 360f;
        while (d < -180f) d += 360f;
        return d;
    }

    public static final class ObsContext {
        public long tick;
        public float health, hunger, absorption;
        public float velX, velY, velZ;
        public float yaw, pitch;
        public float attackCooldown;
        public boolean onGround, sprinting, sneaking, jumping;
        public float fallDistance;
        public boolean isInWater, isOnFire;
        public boolean hasTarget;
        public float tDx, tDy, tDz;
        public float tVelX, tVelY, tVelZ;
        public float targetHealth;
        public float targetHurtTime;
        public float targetAttackCooldown;
        public boolean targetSprinting, targetOnGround, targetSneaking;
        public boolean isLookingAtTarget;
        public float blockUnderFeet;
        public boolean inLiquid;
    }
}