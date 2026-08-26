package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.item.ShieldItem;
import net.minecraft.registry.tag.ItemTags;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;
import net.minecraft.world.World;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Builds the 229-float observation from real Minecraft 1.21.11 client state.
 * Yarn mappings (getYaw/getPitch/getVelocity style names as used by Fabric Yarn 1.21.x).
 *
 * If a mapping differs on your exact yarn build, adjust only the accessors here.
 */
public final class ObservationEncoder {
    private int ticksSinceSwing;
    private int ticksSinceHurt;
    private int ticksSinceTargetHurt;
    private int ticksTargetVisible;
    private int ticksNoTarget;
    private int episodeTick;

    private float prevYawActionNorm;
    private float prevPitchActionNorm;
    private float prevMoveNorm;
    private float prevJump, prevSprint, prevAttack, prevSneak;
    private float lastRewardClip;
    private float rollingAimError;

    /** Network id of last primary target, or -1 if none. Used to gate reward delta. */
    private int lastPrimaryId = -1;
    private int currentPrimaryId = -1;
    private PlayerEntity currentPrimary;

    private final float[] out = new float[RLConstants.OBSERVATION_SIZE];

    public float[] encode(MinecraftClient client) {
        java.util.Arrays.fill(out, 0f);
        ClientPlayerEntity self = client.player;
        World world = client.world;
        if (self == null || world == null) {
            RLConstants.assertObsSize(out.length);
            return out;
        }

        episodeTick++;
        ticksSinceSwing++;
        ticksSinceHurt++;
        ticksSinceTargetHurt++;
        if (self.handSwinging) ticksSinceSwing = 0;
        if (self.hurtTime > 0) ticksSinceHurt = 0;

        encodeSelf(self);
        List<PlayerEntity> others = findOthers(world, self);
        PlayerEntity primary = others.isEmpty() ? null : others.get(0);
        lastPrimaryId = currentPrimaryId;
        currentPrimary = primary;
        currentPrimaryId = primary == null ? -1 : primary.getId();
        encodePrimaryTarget(self, primary, world);
        encodeNearby(self, others, world);
        encodeTiming(primary);
        encodeItems(self);
        encodeTerrain(self, world);
        encodeMisc(self, world, others.size());

        for (int i = 0; i < out.length; i++) {
            out[i] = MathUtil.sane(out[i]);
        }
        RLConstants.assertObsSize(out.length);
        return out;
    }

    public void onAction(ActionSample a, float reward, float aimError) {
        prevMoveNorm = a.move / (float) (RLConstants.MOVE_ACTIONS - 1);
        prevYawActionNorm = a.yawBucket / (float) (RLConstants.YAW_BUCKETS - 1);
        prevPitchActionNorm = a.pitchBucket / (float) (RLConstants.PITCH_BUCKETS - 1);
        prevJump = a.jump ? 1f : 0f;
        prevSprint = a.sprint ? 1f : 0f;
        prevAttack = a.attack ? 1f : 0f;
        prevSneak = a.sneak ? 1f : 0f;
        lastRewardClip = MathUtil.clamp(reward, -1f, 1f);
        rollingAimError = rollingAimError * 0.95f + aimError * 0.05f;
    }

    public void resetEpisode() {
        episodeTick = 0;
        ticksSinceSwing = 0;
        ticksSinceHurt = 0;
        ticksSinceTargetHurt = 0;
        ticksTargetVisible = 0;
        ticksNoTarget = 0;
        rollingAimError = 0f;
        lastPrimaryId = -1;
        currentPrimaryId = -1;
        currentPrimary = null;
    }

    private void encodeSelf(ClientPlayerEntity p) {
        int b = RLConstants.SELF_BASE;
        out[b] = p.getHealth() / 20f;
        out[b + 1] = p.getHungerManager().getFoodLevel() / 20f;
        out[b + 2] = p.getHungerManager().getSaturationLevel() / 20f;
        out[b + 3] = p.getAir() / (float) p.getMaxAir();
        out[b + 4] = p.getAbsorptionAmount() / 20f;
        out[b + 5] = p.getArmor() / 20f;
        out[b + 6] = p.experienceProgress;
        out[b + 7] = p.isOnGround() ? 1f : 0f;
        out[b + 8] = p.isSprinting() ? 1f : 0f;
        out[b + 9] = p.isSneaking() ? 1f : 0f;
        out[b + 10] = p.isSwimming() ? 1f : 0f;
        out[b + 11] = p.isClimbing() ? 1f : 0f;
        out[b + 12] = p.isTouchingWater() ? 1f : 0f;
        out[b + 13] = p.isInLava() ? 1f : 0f;
        out[b + 14] = p.isOnFire() ? 1f : 0f;
        out[b + 15] = p.isUsingItem() ? 1f : 0f;
        out[b + 16] = p.handSwinging ? 1f : 0f;
        out[b + 17] = p.getAttackCooldownProgress(0.5f);
        out[b + 18] = MathUtil.clamp((float) p.fallDistance / 20f, 0f, 1f);

        Vec3d v = p.getVelocity();
        out[b + 19] = MathUtil.clamp((float) v.x / 0.5f, -1f, 1f);
        out[b + 20] = MathUtil.clamp((float) v.y / 0.5f, -1f, 1f);
        out[b + 21] = MathUtil.clamp((float) v.z / 0.5f, -1f, 1f);

        float yaw = p.getYaw();
        float pitch = p.getPitch();
        float body = p.getBodyYaw();
        out[b + 22] = MathHelper.sin(yaw * MathHelper.RADIANS_PER_DEGREE);
        out[b + 23] = MathHelper.cos(yaw * MathHelper.RADIANS_PER_DEGREE);
        out[b + 24] = pitch / 90f;
        out[b + 25] = MathHelper.sin(body * MathHelper.RADIANS_PER_DEGREE);
        out[b + 26] = MathHelper.cos(body * MathHelper.RADIANS_PER_DEGREE);
        out[b + 27] = p.isAlive() ? 1f : 0f;
        out[b + 28] = MathUtil.clamp(p.hurtTime / 10f, 0f, 1f);
        out[b + 29] = MathUtil.clamp(p.getFrozenTicks() / 140f, 0f, 1f);
        out[b + 30] = p.isGliding() ? 1f : 0f;
        out[b + 31] = p.hasVehicle() ? 1f : 0f;
    }

    private void encodePrimaryTarget(ClientPlayerEntity self, PlayerEntity t, World world) {
        int b = RLConstants.TARGET_BASE;
        if (t == null) {
            ticksNoTarget++;
            ticksTargetVisible = 0;
            return;
        }
        ticksNoTarget = 0;
        ticksTargetVisible++;
        if (t.hurtTime > 0) ticksSinceTargetHurt = 0;

        Vec3d eye = self.getEyePos();
        Vec3d tEye = t.getEyePos();
        double dx = tEye.x - eye.x;
        double dy = tEye.y - eye.y;
        double dz = tEye.z - eye.z;
        double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        double hdist = Math.sqrt(dx * dx + dz * dz);

        float desiredYaw = (float) (MathHelper.atan2(dz, dx) * MathHelper.DEGREES_PER_RADIAN) - 90f;
        float desiredPitch = (float) -(MathHelper.atan2(dy, hdist) * MathHelper.DEGREES_PER_RADIAN);
        float yawErr = MathUtil.yawDeltaDeg(self.getYaw(), desiredYaw);
        float pitchErr = desiredPitch - self.getPitch();

        out[b] = 1f;
        out[b + 1] = MathUtil.clamp((float) dx / 32f, -1f, 1f);
        out[b + 2] = MathUtil.clamp((float) dy / 16f, -1f, 1f);
        out[b + 3] = MathUtil.clamp((float) dz / 32f, -1f, 1f);
        out[b + 4] = MathUtil.clamp((float) dist / 32f, 0f, 1f);
        out[b + 5] = MathUtil.clamp(yawErr / 180f, -1f, 1f);
        out[b + 6] = MathUtil.clamp(pitchErr / 90f, -1f, 1f);

        Vec3d tv = t.getVelocity();
        out[b + 7] = MathUtil.clamp((float) tv.x / 0.5f, -1f, 1f);
        out[b + 8] = MathUtil.clamp((float) tv.y / 0.5f, -1f, 1f);
        out[b + 9] = MathUtil.clamp((float) tv.z / 0.5f, -1f, 1f);
        out[b + 10] = t.getHealth() / 20f;
        out[b + 11] = t.isOnGround() ? 1f : 0f;
        out[b + 12] = t.isSprinting() ? 1f : 0f;
        out[b + 13] = t.isSneaking() ? 1f : 0f;
        out[b + 14] = t.handSwinging ? 1f : 0f;
        out[b + 15] = MathHelper.sin(t.getYaw() * MathHelper.RADIANS_PER_DEGREE);
        out[b + 16] = MathHelper.cos(t.getYaw() * MathHelper.RADIANS_PER_DEGREE);
        out[b + 17] = t.getPitch() / 90f;

        boolean los = hasLineOfSight(world, self, eye, tEye);
        out[b + 18] = los ? 1f : 0f;
        out[b + 19] = MathUtil.clamp((float) hdist / 32f, 0f, 1f);
        out[b + 20] = MathUtil.clamp((float) dy / 8f, -1f, 1f);

        Vec3d relVel = tv.subtract(self.getVelocity());
        double closing = 0;
        if (dist > 1e-4) {
            closing = -(relVel.x * dx + relVel.y * dy + relVel.z * dz) / dist;
        }
        out[b + 21] = MathUtil.clamp((float) closing / 0.5f, -1f, 1f);
        out[b + 22] = MathUtil.clamp(t.hurtTime / 10f, 0f, 1f);
        out[b + 23] = 1f; // player
        out[b + 24] = t.getArmor() / 20f;

        // how much target faces us
        float toUsYaw = (float) (MathHelper.atan2(-dx, -dz) * MathHelper.DEGREES_PER_RADIAN);
        float face = MathHelper.cos(MathUtil.yawDeltaDeg(t.getYaw(), toUsYaw) * MathHelper.RADIANS_PER_DEGREE);
        out[b + 25] = face;
        out[b + 26] = t.isUsingItem() ? 1f : 0f;
        out[b + 27] = t.isBlocking() ? 1f : 0f;

        float aimErr = (float) Math.sqrt((yawErr / 180f) * (yawErr / 180f) + (pitchErr / 90f) * (pitchErr / 90f));
        out[b + 28] = MathUtil.clamp(aimErr, 0f, 1f);
        out[b + 29] = dist < RLConstants.MELEE_RANGE ? 1f : 0f;
        out[b + 30] = dist < 6.0 ? 1f : 0f;
        // 63 reserved
    }

    private void encodeNearby(ClientPlayerEntity self, List<PlayerEntity> others, World world) {
        for (int slot = 0; slot < RLConstants.NEARBY_SLOTS; slot++) {
            int base = RLConstants.NEARBY_BASE + slot * RLConstants.NEARBY_STRIDE;
            int idx = slot + 1; // primary is 0
            if (idx >= others.size()) continue;
            PlayerEntity t = others.get(idx);
            Vec3d eye = self.getEyePos();
            Vec3d tEye = t.getEyePos();
            double dx = tEye.x - eye.x;
            double dy = tEye.y - eye.y;
            double dz = tEye.z - eye.z;
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            double hdist = Math.sqrt(dx * dx + dz * dz);
            float desiredYaw = (float) (MathHelper.atan2(dz, dx) * MathHelper.DEGREES_PER_RADIAN) - 90f;
            float desiredPitch = (float) -(MathHelper.atan2(dy, hdist) * MathHelper.DEGREES_PER_RADIAN);
            float yawErr = MathUtil.yawDeltaDeg(self.getYaw(), desiredYaw);
            float pitchErr = desiredPitch - self.getPitch();
            Vec3d tv = t.getVelocity();

            out[base] = 1f;
            out[base + 1] = MathUtil.clamp((float) dx / 32f, -1f, 1f);
            out[base + 2] = MathUtil.clamp((float) dy / 16f, -1f, 1f);
            out[base + 3] = MathUtil.clamp((float) dz / 32f, -1f, 1f);
            out[base + 4] = MathUtil.clamp((float) dist / 32f, 0f, 1f);
            out[base + 5] = MathUtil.clamp(yawErr / 180f, -1f, 1f);
            out[base + 6] = MathUtil.clamp(pitchErr / 90f, -1f, 1f);
            out[base + 7] = MathUtil.clamp((float) tv.x / 0.5f, -1f, 1f);
            out[base + 8] = MathUtil.clamp((float) tv.y / 0.5f, -1f, 1f);
            out[base + 9] = MathUtil.clamp((float) tv.z / 0.5f, -1f, 1f);
            out[base + 10] = t.getHealth() / 20f;
            out[base + 11] = t.isOnGround() ? 1f : 0f;
            out[base + 12] = t.isSprinting() ? 1f : 0f;
            out[base + 13] = t.isSneaking() ? 1f : 0f;
            out[base + 14] = MathHelper.sin(t.getYaw() * MathHelper.RADIANS_PER_DEGREE);
            out[base + 15] = MathHelper.cos(t.getYaw() * MathHelper.RADIANS_PER_DEGREE);
            out[base + 16] = t.getPitch() / 90f;
            out[base + 17] = hasLineOfSight(world, self, eye, tEye) ? 1f : 0f;
            out[base + 18] = MathUtil.clamp((float) hdist / 32f, 0f, 1f);
            out[base + 19] = MathUtil.clamp(t.hurtTime / 10f, 0f, 1f);
            out[base + 20] = 1f;
        }
    }

    private void encodeTiming(PlayerEntity primary) {
        int b = RLConstants.TIMING_BASE;
        out[b] = MathUtil.clamp(ticksSinceSwing / 40f, 0f, 1f);
        out[b + 1] = MathUtil.clamp(ticksSinceHurt / 40f, 0f, 1f);
        out[b + 2] = MathUtil.clamp(ticksSinceTargetHurt / 40f, 0f, 1f);
        out[b + 3] = MathUtil.clamp(ticksTargetVisible / 100f, 0f, 1f);
        out[b + 4] = MathUtil.clamp(ticksNoTarget / 100f, 0f, 1f);
        out[b + 5] = MathUtil.clamp(episodeTick / 2000f, 0f, 1f);
        out[b + 6] = prevYawActionNorm;
        out[b + 7] = prevPitchActionNorm;
        out[b + 8] = prevMoveNorm;
        out[b + 9] = prevJump;
        out[b + 10] = prevSprint;
        out[b + 11] = prevAttack;
        out[b + 12] = prevSneak;
        out[b + 13] = lastRewardClip;
        out[b + 14] = MathUtil.clamp(rollingAimError, 0f, 1f);
    }

    private void encodeItems(ClientPlayerEntity p) {
        int b = RLConstants.ITEM_BASE;
        ItemStack main = p.getMainHandStack();
        ItemStack off = p.getOffHandStack();
        out[b] = main.isIn(ItemTags.SWORDS) ? 1f : 0f;
        out[b + 1] = main.isIn(ItemTags.AXES) ? 1f : 0f;
        out[b + 2] = main.isEmpty() ? 1f : 0f;
        out[b + 3] = p.getAttackCooldownProgress(0.5f);
        out[b + 4] = off.isEmpty() ? 0f : 1f;
        out[b + 5] = off.getItem() instanceof ShieldItem ? 1f : 0f;
        for (int i = 0; i < 9; i++) {
            out[b + 6 + i] = p.getInventory().getStack(i).isEmpty() ? 0f : 1f;
        }
        out[b + 15] = p.getInventory().getSelectedSlot() / 8f;
    }

    private void encodeTerrain(ClientPlayerEntity p, World world) {
        int b = RLConstants.TERRAIN_BASE;
        BlockPos feet = p.getBlockPos();
        out[b] = solid(world, feet.down()) ? 1f : 0f;
        out[b + 1] = solid(world, feet.down(2)) ? 1f : 0f;

        float yaw = p.getYaw() * MathHelper.RADIANS_PER_DEGREE;
        int fx = Math.round(-MathHelper.sin(yaw));
        int fz = Math.round(MathHelper.cos(yaw));
        int lx = Math.round(-MathHelper.sin(yaw - (float) Math.PI / 2));
        int lz = Math.round(MathHelper.cos(yaw - (float) Math.PI / 2));

        out[b + 2] = solid(world, feet.add(fx, 0, fz)) ? 1f : 0f;
        out[b + 3] = solid(world, feet.add(-fx, 0, -fz)) ? 1f : 0f;
        out[b + 4] = solid(world, feet.add(lx, 0, lz)) ? 1f : 0f;
        out[b + 5] = solid(world, feet.add(-lx, 0, -lz)) ? 1f : 0f;
        out[b + 6] = solid(world, feet.add(fx, 1, fz)) ? 1f : 0f;
        out[b + 7] = solid(world, feet.add(-fx, 1, -fz)) ? 1f : 0f;
        out[b + 8] = solid(world, feet.add(lx, 1, lz)) ? 1f : 0f;
        out[b + 9] = solid(world, feet.add(-lx, 1, -lz)) ? 1f : 0f;
        out[b + 10] = solid(world, feet.up(2)) ? 1f : 0f;
        out[b + 11] = solid(world, feet.add(fx, 0, fz)) ? 0f : 1f;

        // 8-dir mid probes
        for (int i = 0; i < 8; i++) {
            float ang = i * (float) Math.PI / 4f;
            int dx = Math.round(-MathHelper.sin(ang));
            int dz = Math.round(MathHelper.cos(ang));
            out[b + 12 + i] = solid(world, feet.add(dx, 0, dz)) ? 1f : 0f;
        }
    }

    private void encodeMisc(ClientPlayerEntity p, World world, int nearCount) {
        int b = RLConstants.MISC_BASE;
        long time = world.getTimeOfDay() % 24000L;
        float phase = (time / 24000f) * (float) Math.PI * 2f;
        out[b] = MathHelper.sin(phase);
        out[b + 1] = MathHelper.cos(phase);
        out[b + 2] = world.isDay() ? 1f : 0f;
        String path = world.getRegistryKey().getValue().getPath();
        out[b + 3] = path.contains("overworld") ? 1f : 0f;
        out[b + 4] = path.contains("nether") ? 1f : 0f;
        out[b + 5] = path.contains("end") ? 1f : 0f;
        out[b + 6] = world.getDifficulty().getId() / 3f;
        out[b + 7] = MathUtil.clamp(nearCount / 8f, 0f, 1f);
        Vec3d v = p.getVelocity();
        out[b + 8] = MathUtil.clamp((float) Math.sqrt(v.x * v.x + v.z * v.z) / 0.5f, 0f, 1f);
        out[b + 9] = Math.abs(out[RLConstants.TARGET_BASE + 7]); // rough
        out[b + 10] = Math.abs(out[RLConstants.TARGET_BASE + 5]);
        out[b + 11] = Math.abs(out[RLConstants.TARGET_BASE + 6]);
        out[b + 12] = 1f; // bias
    }

    private static boolean solid(World world, BlockPos pos) {
        return !world.getBlockState(pos).getCollisionShape(world, pos).isEmpty();
    }

    private static boolean hasLineOfSight(World world, Entity viewer, Vec3d from, Vec3d to) {
        BlockHitResult hit = world.raycast(new RaycastContext(
                from, to,
                RaycastContext.ShapeType.COLLIDER,
                RaycastContext.FluidHandling.NONE,
                viewer));
        return hit.getType() == HitResult.Type.MISS;
    }

    private static List<PlayerEntity> findOthers(World world, PlayerEntity self) {
        List<PlayerEntity> list = new ArrayList<>();
        for (PlayerEntity p : world.getPlayers()) {
            if (p == self || !p.isAlive()) continue;
            if (self.distanceTo(p) > RLConstants.TARGET_RANGE) continue;
            list.add(p);
        }
        list.sort(Comparator.comparingDouble(self::distanceTo));
        return list;
    }

    /** Current aim error [0,1] from last encode (uses primary target fields). */
    public float currentAimError() {
        if (out[RLConstants.TARGET_BASE] < 0.5f) return 1f;
        return MathUtil.clamp(out[RLConstants.TARGET_BASE + 28], 0f, 1f);
    }

    public boolean hasTarget() {
        return out[RLConstants.TARGET_BASE] > 0.5f;
    }

    /** True when this tick's primary is the same entity as last tick's. */
    public boolean sameTargetAsPrevious() {
        return currentPrimaryId >= 0 && currentPrimaryId == lastPrimaryId;
    }

    public int currentTargetId() {
        return currentPrimaryId;
    }

    /**
     * Recompute aim error after camera was moved (post-action).
     * Uses live player yaw/pitch vs last primary target eye. Returns 1 if no target.
     */
    public float aimErrorAfterCamera(ClientPlayerEntity self) {
        PlayerEntity t = currentPrimary;
        if (self == null || t == null || !t.isAlive()) return 1f;
        Vec3d eye = self.getEyePos();
        Vec3d tEye = t.getEyePos();
        double dx = tEye.x - eye.x;
        double dy = tEye.y - eye.y;
        double dz = tEye.z - eye.z;
        double hdist = Math.sqrt(dx * dx + dz * dz);
        float desiredYaw = (float) (MathHelper.atan2(dz, dx) * MathHelper.DEGREES_PER_RADIAN) - 90f;
        float desiredPitch = (float) -(MathHelper.atan2(dy, hdist) * MathHelper.DEGREES_PER_RADIAN);
        float yawErr = MathUtil.yawDeltaDeg(self.getYaw(), desiredYaw);
        float pitchErr = desiredPitch - self.getPitch();
        float aimErr = (float) Math.sqrt(
                (yawErr / 180f) * (yawErr / 180f) + (pitchErr / 90f) * (pitchErr / 90f));
        return MathUtil.clamp(aimErr, 0f, 1f);
    }

    public float[] last() {
        return out;
    }
}