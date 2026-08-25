package com.example.newgen6.client;

import com.example.newgen6.rl.env.ActionSpace;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;

/**
 * Applies the non-movement parts of a sampled PPO action:
 *   - aim: a RELATIVE yaw/pitch delta (mouse-style), via
 *     ClientPlayerEntity#changeLookDirection(deltaYaw, deltaPitch) - the
 *     same entry point vanilla's mouse handler uses. This is explicitly NOT
 *     "yaw = enemyYaw" (forbidden by spec section 18/21); the policy only
 *     ever outputs a bucketed relative turn amount.
 *   - sprint toggle: ClientPlayerEntity#setSprinting(boolean), same call
 *     vanilla issues when the sprint key is pressed.
 *   - attack: routed through MinecraftClient's normal
 *     interactionManager.attackEntity(...) + swingHand(...) pair, i.e. the
 *     exact call sequence vanilla's left-click handling performs. No damage
 *     values, cooldowns, reach, or hitboxes are touched directly - the
 *     server/vanilla combat resolves the result (spec section 23).
 *
 * NOTE: verify attackEntity's exact signature against your 1.21.11 mappings;
 * it has historically lived on ClientPlayerInteractionManager.
 */
public final class AiInputApplier {
    private AiInputApplier() {}

    public static void applyAim(ClientPlayerEntity player, int yawBucket, int pitchBucket) {
        float deltaYaw = ActionSpace.yawBucketToDeltaDegrees(yawBucket);
        float deltaPitch = ActionSpace.pitchBucketToDeltaDegrees(pitchBucket);
        // changeLookDirection applies a RELATIVE turn, exactly like mouse delta input.
        player.changeLookDirection(deltaYaw, deltaPitch);
    }

    public static void applySprint(ClientPlayerEntity player, boolean sprint) {
        player.setSprinting(sprint);
    }

    /** Returns true if an attack was actually issued this tick (cooldown/target permitting). */
    public static boolean applyAttack(ClientPlayerEntity player, LivingEntity target) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.interactionManager == null || target == null) return false;
        if (player.getAttackCooldownProgress(0f) < 1.0f) return false; // respect real cooldown, no bypass

        double reach = 3.0; // matches OPP_IN_ATTACK_RANGE threshold in ObservationSchema
        if (player.squaredDistanceTo(target) > reach * reach) return false;

        mc.interactionManager.attackEntity(player, target);
        player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
        return true;
    }
}