package com.example.newgen6.client;

import com.example.newgen6.rl.*;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Box;

import java.util.List;

/**
 * Ties everything together. Call ClientTickController.register() from your
 * client entrypoint (Newgen6Client#onInitializeClient), after
 * BotKeybinds.register() and BotHud.register().
 */
public class ClientTickController {

    private static final double TARGET_SEARCH_RADIUS = 24.0;

    private static final PPOBrain brain = new PPOBrain();
    private static final RolloutBuffer buffer = new RolloutBuffer();

    private static AIState state = AIState.OFF;
    private static BotAction lastAction = BotAction.NO_OP;

    private static BotState prevState = null;
    private static ActionSample prevSample = null;

    public static void register() {
        ClientTickEvents.END_CLIENT_TICK.register(ClientTickController::onTick);

        // Load any previously saved weights on startup.
        ClientLifecycleEvents.CLIENT_STARTED.register(client -> {
            boolean loaded = WeightPersistence.load(MinecraftClient.getInstance().runDirectory.toPath().resolve("config"), brain);
            if (loaded) {
                System.out.println("[newgen6] Loaded saved PPO weights.");
            } else {
                System.out.println("[newgen6] No saved weights found, starting fresh.");
            }
        });

        // Persist weights on clean shutdown so training carries over between sessions.
        ClientLifecycleEvents.CLIENT_STOPPING.register(client ->
                WeightPersistence.save(client.runDirectory.toPath().resolve("config"), brain));
    }

    private static void onTick(MinecraftClient client) {
        handleKeybinds(client);

        if (state == AIState.OFF) return;
        if (client.player == null || client.world == null) return;

        ClientPlayerEntity player = client.player;
        LivingEntity target = findClosestTarget(client, player);

        BotState current = perceive(client, player, target);
        boolean explore = (state == AIState.TRAINING);
        ActionSample sample = brain.selectAction(current, explore);

        lastAction = sample.action();
        PlayerActionHandler.apply(client, sample.action(), target);

        if (state == AIState.TRAINING && prevState != null && prevSample != null) {
            double reward = computeReward(prevState, current);
            boolean done = (target == null); // treat losing the target as episode boundary
            buffer.add(prevState.toVector(), prevSample.actionIndex(), prevSample.logProb(),
                    reward, prevSample.value(), done);
            brain.maybeTrainAsync(buffer);
        }

        prevState = current;
        prevSample = sample;
    }

    private static void handleKeybinds(MinecraftClient client) {
        while (BotKeybinds.toggleAiState.wasPressed()) {
            state = state.next();
            prevState = null; // reset transition tracking across state changes
            prevSample = null;
            if (client.player != null) {
                client.player.sendMessage(
                        net.minecraft.text.Text.literal("[newgen6] AI state: " + state), true);
            }
        }
        while (BotKeybinds.toggleHud.wasPressed()) {
            BotHud.visible = !BotHud.visible;
        }
    }

    private static LivingEntity findClosestTarget(MinecraftClient client, ClientPlayerEntity player) {
        Box searchBox = player.getBoundingBox().expand(TARGET_SEARCH_RADIUS);
        List<LivingEntity> candidates = client.world.getEntitiesByClass(
                LivingEntity.class, searchBox,
                e -> e != player && e.isAlive() && (e instanceof PlayerEntity)
        );

        LivingEntity closest = null;
        double closestDist = Double.MAX_VALUE;
        for (LivingEntity e : candidates) {
            double d = e.squaredDistanceTo(player);
            if (d < closestDist) {
                closestDist = d;
                closest = e;
            }
        }
        return closest;
    }

    private static BotState perceive(MinecraftClient client, ClientPlayerEntity player, LivingEntity target) {
        double relX = 0, relY = 0, relZ = 0, enemyHealth = 0, los = 0;

        if (target != null) {
            relX = clampNorm((target.getX() - player.getX()) / 16.0);
            relY = clampNorm((target.getY() - player.getY()) / 16.0);
            relZ = clampNorm((target.getZ() - player.getZ()) / 16.0);
            enemyHealth = target.getMaxHealth() > 0 ? target.getHealth() / target.getMaxHealth() : 0;

            // Simple LOS check reusing the same raycast context style as the LIDAR.
            los = hasLineOfSight(client, player, target) ? 1.0 : 0.0;
        }

        double selfHealth = player.getMaxHealth() > 0 ? player.getHealth() / player.getMaxHealth() : 0;
        double velX = player.getVelocity().x;
        double velZ = player.getVelocity().z;
        double cooldown = 1.0 - Math.min(1.0, player.getAttackCooldownProgress(0.0f)); // verify accessor for your version
        double onGround = player.isOnGround() ? 1.0 : 0.0;

        double[] lidar = RaycastLidar.castCone(client.world, player, BotState.LIDAR_RAYS);

        return new BotState(relX, relY, relZ, selfHealth, enemyHealth, velX, velZ, los, cooldown, onGround, lidar);
    }

    private static boolean hasLineOfSight(MinecraftClient client, ClientPlayerEntity player, LivingEntity target) {
        // Reuses the same raycast plumbing as RaycastLidar for a single direct ray to the target.
        net.minecraft.util.math.Vec3d eye = player.getEyePos();
        net.minecraft.util.math.Vec3d targetPos = target.getEyePos();
        net.minecraft.world.RaycastContext ctx = new net.minecraft.world.RaycastContext(
                eye, targetPos,
                net.minecraft.world.RaycastContext.ShapeType.COLLIDER,
                net.minecraft.world.RaycastContext.FluidHandling.NONE,
                player
        );
        net.minecraft.util.hit.HitResult hit = client.world.raycast(ctx);
        return hit == null || hit.getType() == net.minecraft.util.hit.HitResult.Type.MISS;
    }

    private static double clampNorm(double v) {
        return Math.max(-1.0, Math.min(1.0, v));
    }

    private static double computeReward(BotState before, BotState after) {
        double reward = 0.0;
        reward += (before.enemyHealth() - after.enemyHealth()) * 10.0; // damage dealt
        reward -= (before.selfHealth() - after.selfHealth()) * 10.0;   // damage taken
        if (after.enemyHealth() <= 0.0 && before.enemyHealth() > 0.0) reward += 20.0;
        if (after.selfHealth() <= 0.0 && before.selfHealth() > 0.0) reward -= 20.0;
        reward -= 0.01; // small time penalty
        return reward;
    }

    public static PPOBrain getBrain() { return brain; }
    public static AIState getState() { return state; }
    public static BotAction getLastAction() { return lastAction; }
}