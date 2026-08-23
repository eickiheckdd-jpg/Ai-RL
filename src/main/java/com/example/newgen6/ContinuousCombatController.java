package com.example.newgen6;
import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

public class ContinuousCombatController {
    private static final float MAX_AIM_SPEED = 15.0f;

    public static void execute(MinecraftClient client, float[] actionVector) {
        if (client.player == null || client.options == null) return;

        client.player.setYaw(client.player.getYaw() + (actionVector[0] * MAX_AIM_SPEED));
        client.player.setPitch(MathHelper.clamp(client.player.getPitch() + (actionVector[1] * MAX_AIM_SPEED), -90.0f, 90.0f));

        client.options.forwardKey.setPressed(actionVector[2] > 0.2f);
        client.options.backKey.setPressed(actionVector[2] < -0.2f);
        client.options.rightKey.setPressed(actionVector[3] > 0.2f);
        client.options.leftKey.setPressed(actionVector[3] < -0.2f);
        
        client.options.attackKey.setPressed(actionVector[4] > 0.5f);
        client.options.jumpKey.setPressed(actionVector[4] > 0.8f);
    }
}
