package com.example.newgen6;

import net.fabricmc.api.ClientModInitializer;

public class Newgen6Client implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        KeyInputHandler.register();
    }
}
