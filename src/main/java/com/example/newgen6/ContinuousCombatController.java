package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

public class ContinuousCombatController {
    private static final float MAX_AIM_SPEED = 15.0f;

    // --- CURRICULUM TOGGLE ---
    // Set this to PHASE_1_AIM_ONLY to freeze the bot and test its tracking!
    public static CurriculumPhase currentPhase = CurriculumPhase.PHASE_1_AIM_ONLY;

    public enum CurriculumPhase {
        PHASE_1_AIM_ONLY,      // Only Yaw and Pitch are executed
        PHASE_2_FULL_COMBAT    // All 6 action nodes (Movement + Attacks) are executed
    }

    public static void execute(MinecraftClient client, float[] actionVector) {
        if (client.player == null || client.options == null) return;
        
        // Ensure the neural network is outputting the correct array size
        if (actionVector == null || actionVector.length < 5) return;

        // --- 1. ALWAYS ALLOWED: Neck Muscles (Yaw and Pitch) ---
        // actionVector[0] = Yaw (Horizontal), actionVector[1] = Pitch (Vertical)
        client.player.setYaw(client.player.getYaw() + (actionVector[0] * MAX_AIM_SPEED));
        
        // Note: If you want STRICTLY horizontal testing, you can comment out the setPitch line below.
        client.player.setPitch(MathHelper.clamp(client.player.getPitch() + (actionVector[1] * MAX_AIM_SPEED), -90.0f, 90.0f));

        // --- CURRICULUM PHASE 1 ---
        if (currentPhase == CurriculumPhase.PHASE_1_AIM_ONLY) {
            // Force all leg and arm muscles into a neutral, unpressed state.
            client.options.forwardKey.setPressed(false);
            client.options.backKey.setPressed(false);
            client.options.rightKey.setPressed(false);
            client.options.leftKey.setPressed(false);
            client.options.attackKey.setPressed(false);
            client.options.jumpKey.setPressed(false);
            
            // Abort execution here so movement logic below doesn't run
            return; 
        }

        // --- CURRICULUM PHASE 2 (Unlock full combat) ---
        if (currentPhase == CurriculumPhase.PHASE_2_FULL_COMBAT) {
            
            // Actions 2 & 3: Movement Keys (Forward/Back, Right/Left)
            client.options.forwardKey.setPressed(actionVector[2] > 0.2f);
            client.options.backKey.setPressed(actionVector[2] < -0.2f);
            client.options.rightKey.setPressed(actionVector[3] > 0.2f);
            client.options.leftKey.setPressed(actionVector[3] < -0.2f);

            // Action 4: Attack Pulse
            client.options.attackKey.setPressed(actionVector[4] > 0.3f);

            // Action 5: Jump Trigger (Safety check for 6-node setups)
            if (actionVector.length >= 6) {
                client.options.jumpKey.setPressed(actionVector[5] > 0.5f);
            } else {
                client.options.jumpKey.setPressed(actionVector[2] > 0.8f); // Fallback for 5-node setups
            }
        }
    }
}
