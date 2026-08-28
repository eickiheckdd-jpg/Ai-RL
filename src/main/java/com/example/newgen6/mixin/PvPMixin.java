package com.example.newgen6.mixin;

import net.minecraft.client.MinecraftClient;
import org.spongepowered.asm.mixin.Mixin;

@Mixin(MinecraftClient.class)
public class PvPMixin {
    // Reserved for future low-level hooks if needed.
    // Current implementation uses Fabric client tick + HUD events for maximum stability.
}