package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.util.math.MathHelper;

public class ContinuousCombatController {
    private static final float MAX_AIM_SPEED = 15.0f;

    // --- CURRICULUM TOGGLE ---
    // Start with PHASE_1A_HORIZONTAL_AIM so it doesn't get trapped staring at the sky!
    public static CurriculumPhase currentPhase = CurriculumPhase.PHASE_1A_HORIZONTAL_AIM;

    public enum CurriculumPhase {
        PHASE_1A_HORIZONTAL_AIM, // Only Yaw is executed, Pitch is locked flat at 0.0
        PHASE_1B_FULL_AIM,       // Yaw and Pitch are both executed
        PHASE_2_FULL_COMBAT      // All 6 action nodes (Movement + Attacks) are executed
    }

    public static void execute(MinecraftClient client, float[] actionVector) {
        if (client.player == null || client.options == null) return;

        // Ensure the neural network is outputting the correct array size
        if (actionVector == null || actionVector.length < 5) return;

        // --- 1. HORIZONTAL AIM (Always Allowed) ---
        // actionVector[0] = Yaw (Horizontal left/right)
        client.player.setYaw(client.player.getYaw() + (actionVector[0] * MAX_AIM_SPEED));

        // --- 2. VERTICAL AIM CONTROL ---
        if (currentPhase == CurriculumPhase.PHASE_1A_HORIZONTAL_AIM) {
            // Lock pitch perfectly flat so the AI cannot look at the sky and get penalized
            client.player.setPitch(0.0f); 
        } else {
            // Unlock vertical pitch for Phase 1B and Phase 2
            client.player.setPitch(MathHelper.clamp(client.player.getPitch() + (actionVector[1] * MAX_AIM_SPEED), -90.0f, 90.0f));
        }

        // --- 3. CURRICULUM PHASE 1A & 1B (Aim Only) ---
        if (currentPhase == CurriculumPhase.PHASE_1A_HORIZONTAL_AIM || currentPhase == CurriculumPhase.PHASE_1B_FULL_AIM) {
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

        // --- 4. CURRICULUM PHASE 2 (Unlock full combat) ---
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
