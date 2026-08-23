public static float[] getObservation(MinecraftClient client, LivingEntity target) {
    float[] state = new float[16];
    if (client.player == null) return state;

    if (target != null) {
        state[0] = (float) client.player.distanceTo(target);
        state[4] = target.getHealth() / 20.0f;
        // ... fill other target features
    } else {
        state[0] = 20.0f; // Default max distance
        state[4] = 0.0f;
    }

    state[3] = client.player.getHealth() / 20.0f;
    state[5] = client.player.getAttackCooldownProgress(0.0f); // Weapon Cooldown
    return state;
}
