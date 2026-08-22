package com.example.newgen6;

import com.example.newgen6.client.BotHud;
import com.example.newgen6.client.BotKeybinds;
import com.example.newgen6.client.ClientTickController;
import net.fabricmc.api.ClientModInitializer;

public class Newgen6Client implements ClientModInitializer {

    @Override
    public void onInitializeClient() {
        BotKeybinds.register();
        BotHud.register();
        ClientTickController.register();
    }
}