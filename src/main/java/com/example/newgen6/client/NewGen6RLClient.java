package com.example.newgen6.client;

import com.example.newgen6.hud.RlHudOverlay;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;

public class NewGen6RLClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        KeyHandler.register();
        HudRenderCallback.EVENT.register(new RlHudOverlay());
    }
}
