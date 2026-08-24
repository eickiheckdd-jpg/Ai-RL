package com.example.newgen6;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.Hand;
import net.minecraft.util.math.Vec3d;

public class ActionExecutor {

    public static final float TURN_STEP_DEG = 4.0f; 
    public static final double REACH = 3.0;            
    public static final double AIM_CONE_DEG = 12.0;    

    public void resetMovementKeys(MinecraftClient client) {
        client.options.forwardKey.setPressed(false);
        client.options.backKey.setPressed(false);
        client.options.leftKey.setPressed(false);
        client.options.rightKey.setPressed(false);
    }

    public void apply(MinecraftClient client, ActionType action, PlayerEntity opponent) {
        ClientPlayerEntity self = client.player;
        if (self == null) return;

        resetMovementKeys(client);

        switch (action) {
            case MOVE_FORWARD -> client.options.forwardKey.setPressed(true);
            case MOVE_BACKWARD -> client.options.backKey.setPressed(true);
            case STRAFE_LEFT -> client.options.leftKey.setPressed(true);
            case STRAFE_RIGHT -> client.options.rightKey.setPressed(true);
            case TURN_LEFT -> self.setYaw(self.getYaw() - TURN_STEP_DEG);
            case TURN_RIGHT -> self.setYaw(self.getYaw() + TURN_STEP_DEG);
            case LOOK_UP -> self.setPitch(clampPitch(self.getPitch() - TURN_STEP_DEG));
            case LOOK_DOWN -> self.setPitch(clampPitch(self.getPitch() + TURN_STEP_DEG));
            case JUMP -> { if (self.isOnGround()) self.jump(); }
            case ATTACK -> tryAttack(client, self, opponent);
            case NONE -> { /* do nothing this tick */ }
        }
    }

    private float clampPitch(float p) {
        return Math.max(-90.0f, Math.min(90.0f, p));
    }

    private void tryAttack(MinecraftClient client, ClientPlayerEntity self, PlayerEntity opponent) {
        if (opponent == null) return;

        double dist = self.distanceTo(opponent);
        if (dist > REACH) return; 

        Vec3d look = self.getRotationVec(1.0f).normalize();
        
        // FIX: Replaced getPos() with getX(), getY(), getZ()
        Vec3d oppPos = new Vec3d(opponent.getX(), opponent.getY(), opponent.getZ());
        Vec3d toTarget = oppPos
                .add(0, opponent.getStandingEyeHeight() * 0.5, 0)
                .subtract(self.getEyePos())
                .normalize();
                
        double cos = look.dotProduct(toTarget);
        double angleDeg = Math.toDegrees(Math.acos(Math.max(-1.0, Math.min(1.0, cos))));
        if (angleDeg > AIM_CONE_DEG) return; 

        if (client.interactionManager != null) {
            client.interactionManager.attackEntity(self, opponent);
        }
        self.swingHand(Hand.MAIN_HAND);
    }
}
