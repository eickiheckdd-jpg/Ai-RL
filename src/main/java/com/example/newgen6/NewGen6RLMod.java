package com.example.newgen6;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main mod entry (common). Client logic is in {@link com.example.newgen6.client.NewGen6Client}.
 */
public final class NewGen6RLMod implements ModInitializer {
    public static final String MOD_ID = "newgen6";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("NEWGEN6 v2 RL — 229F × 200T (1.21.11)");
    }
}