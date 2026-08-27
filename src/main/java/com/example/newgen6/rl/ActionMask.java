package com.example.newgen6.rl;

/**
 * Masks illegal / useless actions so the agent does not waste exploration.
 * Called every step before sampling.
 */
public final class ActionMask {

    private final boolean[] mask = new boolean[Config.ACTION_DIM];

    /**
     * @return boolean array – true = action allowed
     */
    public boolean[] compute(FeatureExtractor.ObsContext ctx) {
        // default: everything allowed
        for (int i = 0; i < mask.length; i++) mask[i] = true;

        // cannot attack if cooldown not ready
        if (ctx.attackCooldown < 0.85f) {
            mask[ActionSpace.ATTACK] = false;
            mask[ActionSpace.ATTACK_FWD] = false;
            mask[ActionSpace.ATTACK_BACK] = false;
            mask[ActionSpace.ATTACK_LEFT] = false;
            mask[ActionSpace.ATTACK_RIGHT] = false;
            mask[ActionSpace.ATTACK_JUMP] = false;
            mask[ActionSpace.ATTACK_SPRINT] = false;
            mask[ActionSpace.CRIT_ATTEMPT] = false;
            mask[ActionSpace.STRAFE_LEFT_ATK] = false;
            mask[ActionSpace.STRAFE_RIGHT_ATK] = false;
        }

        // cannot jump if already airborne (simple)
        if (!ctx.onGround) {
            mask[ActionSpace.JUMP] = false;
            mask[ActionSpace.JUMP_FWD] = false;
            mask[ActionSpace.ATTACK_JUMP] = false;
            mask[ActionSpace.CRIT_ATTEMPT] = false;
        }

        // no target → disable pure attack primitives that assume a target
        if (!ctx.hasTarget) {
            mask[ActionSpace.HOLD_DIST] = false;
            mask[ActionSpace.CLOSE_IN] = false;
            mask[ActionSpace.BAIT_BACK] = false;
        }

        // in water → some movement less useful
        if (ctx.isInWater) {
            mask[ActionSpace.SPRINT_FWD] = false;
        }

        return mask;
    }

    /** Apply mask to logits (set illegal actions to very negative) */
    public void apply(float[] logits, boolean[] m) {
        for (int i = 0; i < logits.length && i < m.length; i++) {
            if (!m[i]) logits[i] = -1e9f;
        }
    }
}
