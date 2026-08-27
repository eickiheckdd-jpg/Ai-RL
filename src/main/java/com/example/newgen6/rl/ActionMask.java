package com.example.newgen6.rl;

/**
 * Masks physically impossible actions only — no tactic hardcoding.
 */
public final class ActionMask {

    private final boolean[] mask = new boolean[Config.ACTION_DIM];

    public boolean[] compute(FeatureExtractor.ObsContext ctx) {
        for (int i = 0; i < mask.length; i++) mask[i] = true;

        // cannot usefully attack on cooldown
        if (ctx.attackCooldown < 0.85f) {
            mask[ActionSpace.ATTACK] = false;
            mask[ActionSpace.ATK_FWD] = false;
            mask[ActionSpace.ATK_BACK] = false;
            mask[ActionSpace.ATK_LEFT] = false;
            mask[ActionSpace.ATK_RIGHT] = false;
            mask[ActionSpace.ATK_JUMP] = false;
            mask[ActionSpace.ATK_SPRINT] = false;
            mask[ActionSpace.ATK_SNEAK] = false;
        }

        // jump only on ground
        if (!ctx.onGround) {
            mask[ActionSpace.JUMP] = false;
            mask[ActionSpace.JUMP_FWD] = false;
            mask[ActionSpace.JUMP_LEFT] = false;
            mask[ActionSpace.JUMP_RIGHT] = false;
            mask[ActionSpace.ATK_JUMP] = false;
        }

        if (ctx.isInWater) {
            mask[ActionSpace.SPRINT_FWD] = false;
            mask[ActionSpace.SPRINT_LEFT] = false;
            mask[ActionSpace.SPRINT_RIGHT] = false;
        }

        return mask;
    }

    public void apply(float[] logits, boolean[] m) {
        for (int i = 0; i < logits.length && i < m.length; i++) {
            if (!m[i]) logits[i] = -1e9f;
        }
    }
}