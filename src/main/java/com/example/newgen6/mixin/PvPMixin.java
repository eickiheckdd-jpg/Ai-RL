package com.example.newgen6.mixin;

import com.example.newgen6.client.AiControlState;
import net.minecraft.client.input.KeyboardInput;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * ================================================================
 * MIXIN DOCUMENTATION (spec section 22 requires every injection documented)
 * ================================================================
 * Target class : net.minecraft.client.input.KeyboardInput
 *                (the class that translates raw keyboard state into the
 *                 movementForward / movementSideways / jumping / sneaking
 *                 fields consumed every tick by ClientPlayerEntity).
 *                NOTE: verify this class/field names against your actual
 *                1.21.11 Yarn mappings - it was not possible to confirm
 *                them against your real project sources here.
 * Target method : tick(boolean slowDown, float sprintMultiplier) [signature
 *                 may differ slightly across 1.21.x - verify against your
 *                 mapped sources]
 * Injection point: TAIL (after vanilla has already computed the human
 *                 keyboard-derived input state for this tick)
 * Reason        : This is the ONLY point where we override the input state
 *                 with the AI's chosen action, so that from here on
 *                 Minecraft's normal input pipeline (movement, physics,
 *                 collision, sprint/jump handling) runs completely
 *                 unmodified - satisfying spec section 21 (no teleporting,
 *                 no direct velocity/position writes, no bypassing normal
 *                 combat). We are emulating "the keys are held", not
 *                 puppeting the player entity directly.
 * Thread        : Client render/tick thread (KeyboardInput#tick is only
 *                 ever called from the client tick loop).
 * Data obtained/mutated: this.movementForward, this.movementSideways,
 *                 this.jumping, this.sneaking (inherited from Input) - only
 *                 written when AiControlState.isAiControlEnabled() is true,
 *                 so a human can flip control back at any time.
 *
 * Aim (yaw/pitch) and attack are intentionally NOT handled here: they are
 * applied via ClientPlayerEntity#changeLookDirection(deltaYaw, deltaPitch)
 * and the normal attack-entity interaction path respectively, from
 * com.example.newgen6.client.AiInputApplier, called once per tick from the
 * client tick event in NewGen6RLMod. Both are relative/normal-path
 * operations, never an absolute "snap to target" write.
 */
@Mixin(KeyboardInput.class)
public abstract class PvPMixin {

    @Inject(method = "tick", at = @At("TAIL"))
    private void newgen6$overrideWithAiInput(boolean slowDown, float sprintMultiplier, CallbackInfo ci) {
        if (!AiControlState.isAiControlEnabled()) return;

        KeyboardInput self = (KeyboardInput) (Object) this;
        AiControlState.PendingInput pending = AiControlState.consumePendingMovementInput();
        if (pending == null) return;

        // Overwrite the fields KeyboardInput just computed from real keyboard
        // state with the AI's chosen movement action for this tick. Values
        // are the same field types/ranges vanilla itself would produce.
        self.movementForward = pending.forward;
        self.movementSideways = pending.sideways;
        self.jumping = pending.jump;
        self.sneaking = pending.sneak;
    }
}