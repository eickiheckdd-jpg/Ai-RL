package com.example.newgen6.client;

import com.example.newgen6.rl.ActionSpace;
import com.example.newgen6.rl.FeatureExtractor;
import com.example.newgen6.rl.PPOAgent;
import com.example.newgen6.hud.CombatGui;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;
import org.lwjgl.glfw.GLFW;

/**
 * Client entrypoint + main RL loop + keybinds.
 *
 * X = toggle Debug HUD
 * C = toggle AI on/off
 * F6 = save checkpoint
 * F7 = toggle TRAIN / EVAL
 * F8 = start/stop replay recording
 * F9 = emergency stop / clear
 */
public class NewGen6Client implements ClientModInitializer {

    public static final PPOAgent AGENT = new PPOAgent();

    private static boolean aiEnabled = true;
    private static boolean debugVisible = true;
    private static long tickCounter = 0;

    public static boolean flagHit = false;
    public static boolean flagCrit = false;
    public static boolean flagHurt = false;
    public static float lastDmgDealt = 0f;
    public static float lastDmgTaken = 0f;

    private static KeyBinding keyToggleAI;
    private static KeyBinding keyToggleDebug;
    private static KeyBinding keySave;
    private static KeyBinding keyEval;
    private static KeyBinding keyReplay;
    private static KeyBinding keyEmergency;

    private static final KeyBinding.Category CATEGORY = KeyBinding.Category.create("newgen6");

    @Override
    public void onInitializeClient() {
        keyToggleAI = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_ai", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_C, CATEGORY));
        keyToggleDebug = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.toggle_debug", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_X, CATEGORY));
        keySave = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.save", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_F6, CATEGORY));
        keyEval = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.eval", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_F7, CATEGORY));
        keyReplay = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.replay", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_F8, CATEGORY));
        keyEmergency = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.newgen6.emergency", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_F9, CATEGORY));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (keyToggleAI.wasPressed()) {
                aiEnabled = !aiEnabled;
                System.out.println("[NEWGEN6] AI " + (aiEnabled ? "ENABLED" : "DISABLED"));
            }
            while (keyToggleDebug.wasPressed()) {
                debugVisible = !debugVisible;
                System.out.println("[NEWGEN6] Debug HUD " + (debugVisible ? "ON" : "OFF"));
            }
            while (keySave.wasPressed()) {
                AGENT.getCheckpoints().saveLatest(AGENT);
                AGENT.getCheckpoints().saveVersioned(AGENT);
                System.out.println("[NEWGEN6] Manual save complete");
            }
            while (keyEval.wasPressed()) {
                boolean now = !AGENT.isTraining();
                AGENT.setTraining(now);
                System.out.println("[NEWGEN6] Mode → " + (now ? "TRAINING" : "EVALUATION"));
            }
            while (keyReplay.wasPressed()) {
                if (AGENT.getReplay().isRecording()) AGENT.getReplay().stopAndSave();
                else AGENT.getReplay().start();
            }
            while (keyEmergency.wasPressed()) {
                if (AGENT.isEmergency()) {
                    AGENT.clearEmergency();
                    aiEnabled = true;
                    System.out.println("[NEWGEN6] Emergency cleared");
                } else {
                    AGENT.emergencyStop();
                    aiEnabled = false;
                }
            }
        });

        CombatGui.register();
        System.out.println("[NEWGEN6] Client ready | X=Debug  C=AI  F6=Save  F7=Eval  F8=Replay  F9=Emergency");
    }

    public static void onTick(MinecraftClient mc) {
        if (!aiEnabled || mc.player == null || AGENT.isEmergency()) return;
        tickCounter++;

        ClientPlayerEntity self = mc.player;
        FeatureExtractor.ObsContext ctx = new FeatureExtractor.ObsContext();
        ctx.tick = tickCounter;

        ctx.health = self.getHealth();
        ctx.hunger = self.getHungerManager().getFoodLevel();
        ctx.absorption = self.getAbsorptionAmount();
        Vec3d vel = self.getVelocity();
        ctx.velX = (float) vel.x;
        ctx.velY = (float) vel.y;
        ctx.velZ = (float) vel.z;
        ctx.yaw = self.getYaw();
        ctx.pitch = self.getPitch();
        ctx.attackCooldown = self.getAttackCooldownProgress(0f);
        ctx.onGround = self.isOnGround();
        ctx.sprinting = self.isSprinting();
        ctx.sneaking = self.isSneaking();
        ctx.jumping = !self.isOnGround() && vel.y > 0;
        ctx.fallDistance = (float) self.fallDistance;
        ctx.isInWater = self.isTouchingWater();
        ctx.isOnFire = self.isOnFire();

        PlayerEntity target = findTarget(mc, self);
        if (target != null) {
            ctx.hasTarget = true;
            ctx.tDx = (float) (target.getX() - self.getX());
            ctx.tDy = (float) (target.getY() - self.getY());
            ctx.tDz = (float) (target.getZ() - self.getZ());
            Vec3d tVel = target.getVelocity();
            ctx.tVelX = (float) tVel.x;
            ctx.tVelY = (float) tVel.y;
            ctx.tVelZ = (float) tVel.z;
            ctx.targetHealth = target.getHealth();
            ctx.targetHurtTime = target.hurtTime;
            ctx.targetAttackCooldown = target.getAttackCooldownProgress(0f);
            ctx.targetSprinting = target.isSprinting();
            ctx.targetOnGround = target.isOnGround();
            ctx.targetSneaking = target.isSneaking();
            HitResult hr = mc.crosshairTarget;
            ctx.isLookingAtTarget = hr instanceof EntityHitResult ehr && ehr.getEntity() == target;
        }

        ctx.blockUnderFeet = self.isOnGround() ? 1f : 0f;
        ctx.inLiquid = self.isTouchingWater() || self.isInLava();

        ActionSpace.Control ctrl = AGENT.act(ctx, flagHit, flagCrit, flagHurt, lastDmgDealt, lastDmgTaken);
        flagHit = flagCrit = flagHurt = false;
        lastDmgDealt = lastDmgTaken = 0f;

        InputController.apply(mc, ctrl);
    }

    private static PlayerEntity findTarget(MinecraftClient mc, ClientPlayerEntity self) {
        if (mc.crosshairTarget instanceof EntityHitResult ehr
                && ehr.getEntity() instanceof PlayerEntity pe && pe != self) {
            return pe;
        }
        PlayerEntity nearest = null;
        double best = 36.0;
        Box box = self.getBoundingBox().expand(6.0);
        for (Entity e : mc.world.getOtherEntities(self, box)) {
            if (e instanceof PlayerEntity pe) {
                double d = self.squaredDistanceTo(pe);
                if (d < best) {
                    best = d;
                    nearest = pe;
                }
            }
        }
        return nearest;
    }

    public static boolean isAiEnabled() { return aiEnabled; }
    public static boolean isDebugVisible() { return debugVisible; }
}