package com.example.newgen6.mixin;

import com.example.newgen6.client.AiControlState;
import net.minecraft.client.input.KeyboardInput;
import net.minecraft.util.PlayerInput;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardInput.class)
public abstract class PvPMixin {

    @Inject(method = "tick", at = @At("TAIL"))
    private void newgen6$overrideWithAiInput(CallbackInfo ci) {
        if (!AiControlState.isAiControlEnabled()) return;

        KeyboardInput self = (KeyboardInput) (Object) this;
        AiControlState.PendingInput pending = AiControlState.consumePendingMovementInput();
        if (pending == null) return;

        // Sanitize inputs to ensure mutually exclusive directional flags
        boolean forward = pending.forward > 0f;
        boolean backward = pending.forward < 0f;
        if (forward && backward) {
            forward = false;
            backward = false;
        }

        boolean left = pending.sideways > 0f;
        boolean right = pending.sideways < 0f;
        if (left && right) {
            left = false;
            right = false;
        }

        self.playerInput = new PlayerInput(
            forward,
            backward,
            left,
            right,
            pending.jump,
            pending.sneak,
            false // sprint
        );
    }
}
