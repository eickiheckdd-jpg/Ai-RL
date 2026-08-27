package com.example.newgen6.rl;

/**
 * Strict reward shaping for HT3 Sword.
 *
 * Design goals:
 * - Random noise / spinning / looking away must NOT farm reward
 * - Almost all positive combat reward is gated on facing the enemy
 * - Spacing, timing, and clean aim are heavily preferred
 * - Wasteful mouse thrashing is penalized after stage 0
 */
public final class RewardCalculator {

    private float lastHealth = 20f;
    private float lastTargetHealth = 20f;
    private float lastHDist = 3.1f;
    private int consecutiveHits = 0;
    private int ticksLookingAway = 0;
    private int ticksNoTarget = 0;
    private float smoothAimPenalty = 0f;

    public void reset() {
        lastHealth = 20f;
        lastTargetHealth = 20f;
        lastHDist = 3.1f;
        consecutiveHits = 0;
        ticksLookingAway = 0;
        ticksNoTarget = 0;
        smoothAimPenalty = 0f;
    }

    public float compute(FeatureExtractor.ObsContext ctx, ActionSpace.Control ctrl,
                         boolean didHit, boolean didCrit, boolean didTakeDamage,
                         float dmgDealt, float dmgTaken, int stage) {

        float r = 0f;

        // ============================================================
        // 1. SURVIVAL (always on, small)
        // ============================================================
        float healthDelta = ctx.health - lastHealth;
        if (healthDelta < 0) r += healthDelta * 1.1f;          // taking damage hurts
        if (didTakeDamage) r -= 0.45f;
        if (ctx.health <= 0.5f) r -= 2.0f;                     // near death
        lastHealth = ctx.health;

        // tiny alive bonus only early (so it doesn't farm by standing still later)
        if (stage <= 1) r += 0.008f;

        // ============================================================
        // 2. NO TARGET → almost no positive reward possible
        // ============================================================
        if (!ctx.hasTarget) {
            ticksNoTarget++;
            ticksLookingAway = 0;
            consecutiveHits = 0;
            // mild boredom penalty if wandering forever with no opponent
            if (ticksNoTarget > 40) r -= 0.02f;
            if (ticksNoTarget > 100) r -= 0.05f;
            // random spinning with no target is pure waste
            float mouseMag = Math.abs(ctrl.mouseDx) + Math.abs(ctrl.mouseDy);
            if (mouseMag > 6f) r -= 0.06f * Math.min(mouseMag / 10f, 2f);
            return clip(r);
        }
        ticksNoTarget = 0;

        // ============================================================
        // 3. ANGLE / FACING GATE (core anti-noise)
        // ============================================================
        float hDist = (float) Math.sqrt(ctx.tDx * ctx.tDx + ctx.tDz * ctx.tDz);
        // angle between look direction and vector-to-target (approx from feature)
        // We use isLookingAtTarget + a soft angle proxy from relative features.
        boolean facing = ctx.isLookingAtTarget;
        // soft facing: if horizontal distance is small and we are attacking, still require look
        float faceScore = facing ? 1f : 0f;

        if (!facing) {
            ticksLookingAway++;
            // escalating penalty for staring at the sky / ground / walls
            r -= 0.04f + Math.min(ticksLookingAway, 60) * 0.004f;
            // attacking while not looking = almost always noise
            if (ctrl.attack) r -= 0.35f;
        } else {
            ticksLookingAway = 0;
            r += 0.07f; // small steady bonus for keeping crosshair on target
        }

        // ============================================================
        // 4. DAMAGE / HITS — heavily gated on facing
        // ============================================================
        if (didHit) {
            if (facing) {
                // legitimate hit
                float hitReward = 1.4f + dmgDealt * 0.45f;
                consecutiveHits++;
                hitReward += Math.min(consecutiveHits, 8) * 0.18f; // combo
                if (didCrit) hitReward += 1.1f;
                // bonus if hit while in good spacing
                if (hDist > 2.4f && hDist < 3.6f) hitReward += 0.35f;
                r += hitReward;
            } else {
                // hit somehow without looking — still possible via sweep etc.
                // give almost nothing so random flailing is not profitable
                r += 0.15f + dmgDealt * 0.05f;
                consecutiveHits = 0;
            }
        } else {
            // missed attack while swinging
            if (ctrl.attack && facing && ctx.attackCooldown > 0.9f) {
                // swung when ready and looking but missed — small cost
                r -= 0.08f;
            } else if (ctrl.attack && !facing) {
                r -= 0.25f; // swinging at nothing
            }
            if (!didHit) {
                // decay combo only when we clearly stopped hitting
                if (!ctrl.attack) consecutiveHits = 0;
            }
        }

        // target HP drop (only count if we are engaged / facing)
        float tDelta = lastTargetHealth - ctx.targetHealth;
        if (tDelta > 0.1f) {
            r += facing ? tDelta * 0.55f : tDelta * 0.08f;
        }
        lastTargetHealth = ctx.targetHealth;

        // ============================================================
        // 5. SPACING (only meaningful when we have a target)
        // ============================================================
        float optimal = 3.05f;
        float err = Math.abs(hDist - optimal);

        if (facing) {
            if (err < 0.30f) r += 0.28f;           // sweet spot
            else if (err < 0.55f) r += 0.12f;
            else if (err < 1.0f) r += 0.02f;
            else r -= Math.min(err * 0.10f, 0.45f);
        } else {
            // not looking → spacing reward is almost zero (prevents orbiting while AFK-looking)
            if (err < 0.30f) r += 0.03f;
            else r -= Math.min(err * 0.06f, 0.25f);
        }

        // intelligent close / create space
        float closing = lastHDist - hDist;
        if (facing) {
            if (hDist > 4.2f && closing > 0.02f) r += 0.12f;   // close the gap
            if (hDist < 2.15f && closing < -0.02f) r += 0.14f; // create space
            if (hDist < 1.6f) r -= 0.15f;                       // too close / body cam
        }
        lastHDist = hDist;

        // ============================================================
        // 6. AIM QUALITY / ANTI-SHAKE
        // ============================================================
        float mouseMag = Math.abs(ctrl.mouseDx) + Math.abs(ctrl.mouseDy);

        if (stage == 0) {
            // Neutral: allow exploration, only punish extreme thrashing
            if (mouseMag > 18f) r -= 0.05f;
        } else {
            // After stage 0: smooth aim is required
            if (mouseMag > 10f) r -= 0.07f * Math.min(mouseMag / 12f, 2.5f);
            // micro-adjustments while already facing = good
            if (facing && mouseMag > 0.15f && mouseMag < 4.5f) r += 0.04f;
            // huge flicks while not facing = noise
            if (!facing && mouseMag > 8f) r -= 0.10f;
        }

        // ============================================================
        // 7. ATTACK TIMING (only when facing)
        // ============================================================
        if (facing && ctrl.attack && ctx.attackCooldown >= 0.88f) {
            r += 0.18f; // swung on a ready cooldown while looking
        }
        if (facing && ctrl.attack && ctx.attackCooldown < 0.5f) {
            r -= 0.12f; // panic clicking on cooldown
        }

        // ============================================================
        // 8. STAGE-SPECIFIC EMPHASIS
        // ============================================================
        switch (stage) {
            case 0 -> {
                // Neutral: tiny exploration, still no free lunch for spinning
                if (facing) r += 0.02f;
            }
            case 1 -> {
                // learn to keep crosshair near enemy
                if (facing) r += 0.06f;
                if (mouseMag < 3.5f && facing) r += 0.04f;
            }
            case 2, 3 -> {
                if (facing && err < 0.5f) r += 0.10f;
                if (didHit && facing) r += 0.25f;
            }
            case 4, 5, 6 -> {
                if (didCrit && facing) r += 0.6f;
                if (consecutiveHits >= 3 && facing) r += 0.35f;
                if (consecutiveHits >= 5 && facing) r += 0.25f;
                // punish looking away mid-combo hard
                if (!facing && consecutiveHits > 0) r -= 0.4f;
            }
        }

        // ============================================================
        // 9. FINAL CLIP + SCALE
        // ============================================================
        return clip(r);
    }

    private static float clip(float r) {
        r = Math.max(-4.0f, Math.min(4.0f, r));
        return r * Config.REWARD_SCALE;
    }
}
