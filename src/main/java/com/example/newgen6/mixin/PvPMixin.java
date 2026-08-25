package com.example.newgen6.mixin;

import com.example.newgen6.client.AiControlState;
import net.minecraft.client.input.KeyboardInput;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardInput.class)
public abstract class PvPMixin {

    @Inject(method = "tick", at = @At("TAIL"))
    private void newgen6$overrideWithAiInput(boolean slowDown, float sprintMultiplier, CallbackInfo ci) {
        if (!AiControlState.isAiControlEnabled()) return;

        KeyboardInput self = (KeyboardInput) (Object) this;
        AiControlState.PendingInput pending = AiControlState.consumePendingMovementInput();
        if (pending == null) return;

        self.movementForward = pending.forward;
        self.movementSideways = pending.sideways;
        self.jumping = pending.jump;
        self.sneaking = pending.sneak;
    }
}
