package com.example.newgen6.client;

import com.example.newgen6.rl.ActionSpace;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.MathHelper;

/**
 * Keyboard-style key presses + continuous mouse deltas for aiming.
 * Compatible with Minecraft 1.21.11 / Yarn.
 */
public final class InputController {

    private InputController() {}

    public static void apply(MinecraftClient mc, ActionSpace.Control c) {
        ClientPlayerEntity p = mc.player;
        if (p == null) return;

        // Continuous mouse deltas (not discrete look actions)
        float yaw = p.getYaw() + c.mouseDx;
        float pitch = MathHelper.clamp(p.getPitch() + c.mouseDy, -90f, 90f);
        p.setYaw(yaw);
        p.setPitch(pitch);

        // Release then re-press movement keys
        setKey(mc.options.forwardKey, false);
        setKey(mc.options.backKey, false);
        setKey(mc.options.leftKey, false);
        setKey(mc.options.rightKey, false);
        setKey(mc.options.jumpKey, false);
        setKey(mc.options.sneakKey, false);
        setKey(mc.options.sprintKey, false);

        if (c.forward > 0.3f) setKey(mc.options.forwardKey, true);
        else if (c.forward < -0.3f) setKey(mc.options.backKey, true);
        if (c.strafe > 0.3f) setKey(mc.options.leftKey, true);
        else if (c.strafe < -0.3f) setKey(mc.options.rightKey, true);
        if (c.jump) setKey(mc.options.jumpKey, true);
        if (c.sneak) setKey(mc.options.sneakKey, true);
        if (c.sprint && c.forward > 0.3f) {
            setKey(mc.options.sprintKey, true);
            p.setSprinting(true);
        }

        // Attack without calling private doAttack()
        if (c.attack) {
            HitResult hit = mc.crosshairTarget;
            if (hit instanceof EntityHitResult ehr && mc.interactionManager != null) {
                Entity target = ehr.getEntity();
                mc.interactionManager.attackEntity(p, target);
                p.swingHand(p.getActiveHand());
            } else {
                p.swingHand(p.getActiveHand());
            }
        }
    }

    public static void releaseAll(MinecraftClient mc) {
        if (mc == null || mc.options == null) return;
        setKey(mc.options.forwardKey, false);
        setKey(mc.options.backKey, false);
        setKey(mc.options.leftKey, false);
        setKey(mc.options.rightKey, false);
        setKey(mc.options.jumpKey, false);
        setKey(mc.options.sneakKey, false);
        setKey(mc.options.sprintKey, false);
    }

    private static void setKey(KeyBinding key, boolean pressed) {
        key.setPressed(pressed);
        if (pressed) {
            KeyBinding.onKeyPressed(key.getDefaultKey());
        }
    }
}