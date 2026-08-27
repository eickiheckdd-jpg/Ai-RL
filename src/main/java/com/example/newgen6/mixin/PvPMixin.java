package com.example.newgen6.mixin;

import com.example.newgen6.client.NewGen6Client;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Core client-side hooks for observation extraction and
 * action application on 1.21.11.
 *
 * We inject at the end of the client tick so all vanilla
 * state (velocity, attack cooldown, etc.) is up to date.
 */
@Mixin(MinecraftClient.class)
public class PvPMixin {

    @Inject(method = "tick", at = @At("RETURN"))
    private void newgen6$onClientTick(CallbackInfo ci) {
        MinecraftClient mc = (MinecraftClient) (Object) this;
        if (mc.player == null || mc.world == null) return;
        if (mc.isPaused()) return;

        NewGen6Client.onTick(mc);
    }
}