package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NewGen6RLMod implements ModInitializer {
    public static final String MOD_ID = "newgen6";
    public static final Logger LOGGER = LoggerFactory.getLogger("NEWGEN6");

    @Override
    public void onInitialize() {
        LOGGER.info("[NEWGEN6] Pure-Java RL core loaded");
    }
}