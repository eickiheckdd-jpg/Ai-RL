package com.example.newgen6.mixin;

import com.example.newgen6.client.AiControlState;
import net.minecraft.client.input.KeyboardInput;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardInput.class)
public class PvPMixin {

    @Inject(method = "tick", at = @At("TAIL"))
    private void overrideInput(boolean slowDown, float slowDownFactor, CallbackInfo ci) {
        if (!AiControlState.isAiControlEnabled()) return;

        AiControlState.PendingInput pending = AiControlState.consumePendingMovementInput();
        KeyboardInput self = (KeyboardInput) (Object) this;

        self.movementForward = pending.forward;
        self.movementSideways = pending.sideways;
        self.jumping = pending.jump;
        self.sneaking = pending.sneak;
    }
}
