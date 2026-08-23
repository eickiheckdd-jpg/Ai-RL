package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

public class ContinuousCombatController {
    private static final float MAX_AIM_SPEED = 15.0f;

    public static void execute(MinecraftClient client, float[] actionVector) {
        if (client.player == null || client.options == null) return;
        // Require at least 5 actions (Yaw, Pitch, Forward, Strafe, Attack)
        if (actionVector == null || actionVector.length < 5) return;

        // Actions 0 & 1: Continuous Aim Control (Yaw and Pitch)
        client.player.setYaw(client.player.getYaw() + (actionVector[0] * MAX_AIM_SPEED));
        client.player.setPitch(MathHelper.clamp(client.player.getPitch() + (actionVector[1] * MAX_AIM_SPEED), -90.0f, 90.0f));

        // Actions 2 & 3: Movement Keys (Forward/Back, Right/Left)
        client.options.forwardKey.setPressed(actionVector[2] > 0.2f);
        client.options.backKey.setPressed(actionVector[2] < -0.2f);
        client.options.rightKey.setPressed(actionVector[3] > 0.2f);
        client.options.leftKey.setPressed(actionVector[3] < -0.2f);

        // Action 4: Attack Pulse
        client.options.attackKey.setPressed(actionVector[4] > 0.3f);

        // Action 5: Optional Jump Trigger (Only executes if the network outputs 6 nodes)
        if (actionVector.length >= 6) {
            client.options.jumpKey.setPressed(actionVector[5] > 0.5f);
        } else {
            // Fallback: If 5-node vector, jump when moving forward aggressively
            client.options.jumpKey.setPressed(actionVector[2] > 0.8f);
        }
    }
}
