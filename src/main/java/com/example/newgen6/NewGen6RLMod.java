package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NewGen6RLMod implements ModInitializer {
    public static final String MOD_ID = "newgen6";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("NEWGEN6 RL initialized — pure Java PPO targeting HT3 Sword on 1.21.11");
    }
}