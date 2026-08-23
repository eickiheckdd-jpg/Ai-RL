package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

public class ContinuousCombatController {
    private static final float MAX_AIM_SPEED = 15.0f;

    public static void execute(MinecraftClient client, float[] actionVector) {
        if (client.player == null || client.options == null) return;
        if (actionVector == null || actionVector.length < 6) return;

        // Action 0 & 1: Continuous Aim Control (Yaw and Pitch Deltas)
        client.player.setYaw(client.player.getYaw() + (actionVector[0] * MAX_AIM_SPEED));
        client.player.setPitch(MathHelper.clamp(client.player.getPitch() + (actionVector[1] * MAX_AIM_SPEED), -90.0f, 90.0f));

        // Action 2 & 3: Movement Keys (Forward/Back, Right/Left)
        client.options.forwardKey.setPressed(actionVector[2] > 0.2f);
        client.options.backKey.setPressed(actionVector[2] < -0.2f);
        client.options.rightKey.setPressed(actionVector[3] > 0.2f);
        client.options.leftKey.setPressed(actionVector[3] < -0.2f);

        // Action 4: Attack Pulse (Triggers sword swing when output > 0.3)
        client.options.attackKey.setPressed(actionVector[4] > 0.3f);

        // Action 5: Jump Trigger (Independent movement node > 0.5)
        client.options.jumpKey.setPressed(actionVector[5] > 0.5f);
    }
}
