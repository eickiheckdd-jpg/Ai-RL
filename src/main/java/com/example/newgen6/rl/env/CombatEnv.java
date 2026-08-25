package com.example.newgen6.rl.env;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;

public final class CombatEnv {

    private LivingEntity currentOpponent;
    private int opponentTier = 1;

    private int ticksSinceOwnAttack = 999;
    private int ticksSinceHitLanded = 999;
    private int ticksSinceDamageTaken = 999;
    private float recentDamageDealt = 0f;
    private float recentDamageReceived = 0f;
    private boolean lastAttackHit = false;
    private boolean lastAttackMissed = false;
    private int ticksThisEpisode = 0;

    private int prevMoveAction = ActionSpace.MOVE_HOLD;
    private boolean prevJump, prevSprint, prevSneak, prevAttack;
    private float prevYawDeltaDeg, prevPitchDeltaDeg;

    public void setOpponent(LivingEntity opponent) { this.currentOpponent = opponent; }
    public void setOpponentTier(int tier) { this.opponentTier = tier; }

    public void recordAction(int moveAction, boolean jump, boolean sprint, boolean sneak, boolean attack,
                              float yawDeltaDeg, float pitchDeltaDeg) {
        prevMoveAction = moveAction; prevJump = jump; prevSprint = sprint; prevSneak = sneak; prevAttack = attack;
        prevYawDeltaDeg = yawDeltaDeg; prevPitchDeltaDeg = pitchDeltaDeg;
        if (attack) ticksSinceOwnAttack = 0; else ticksSinceOwnAttack = Math.min(999, ticksSinceOwnAttack + 1);
    }

    public void onDamageDealt(float amount, boolean hit) {
        if (hit) { recentDamageDealt += amount; ticksSinceHitLanded = 0; lastAttackHit = true; lastAttackMissed = false; }
        else { lastAttackMissed = true; lastAttackHit = false; }
    }

    public void onDamageTaken(float amount) {
        recentDamageReceived += amount;
        ticksSinceDamageTaken = 0;
    }

    public void tickCounters() {
        ticksThisEpisode++;
        ticksSinceHitLanded = Math.min(999, ticksSinceHitLanded + 1);
        ticksSinceDamageTaken = Math.min(999, ticksSinceDamageTaken + 1);
        recentDamageDealt *= 0.95f;
        recentDamageReceived *= 0.95f;
    }

    public void resetEpisode() {
        ticksThisEpisode = 0;
        ticksSinceOwnAttack = ticksSinceHitLanded = ticksSinceDamageTaken = 999;
        recentDamageDealt = recentDamageReceived = 0f;
    }

    public void collect(float[] obsOut) {
        MinecraftClient mc = MinecraftClient.getInstance();
        ClientPlayerEntity self = mc.player;
        if (self == null) { java.util.Arrays.fill(obsOut, 0f); return; }

        java.util.Arrays.fill(obsOut, 0f);

        obsOut[ObservationSchema.SELF_HEALTH_FRACTION] = clamp01(self.getHealth() / Math.max(1f, self.getMaxHealth()));
        obsOut[ObservationSchema.SELF_ABSORPTION_FRACTION] = clamp01(self.getAbsorptionAmount() / Math.max(1f, self.getMaxHealth()));
        Vec3d vel = self.getVelocity();
        obsOut[ObservationSchema.SELF_VEL_X] = clampSigned((float) vel.x / ObservationSchema.MAX_SPEED);
        obsOut[ObservationSchema.SELF_VEL_Y] = clampSigned((float) vel.y / ObservationSchema.MAX_SPEED);
        obsOut[ObservationSchema.SELF_VEL_Z] = clampSigned((float) vel.z / ObservationSchema.MAX_SPEED);
        float horizSpeed = (float) Math.sqrt(vel.x * vel.x + vel.z * vel.z);
        obsOut[ObservationSchema.SELF_HORIZ_SPEED] = clamp01(horizSpeed / ObservationSchema.MAX_SPEED);
        obsOut[ObservationSchema.SELF_VERT_SPEED] = clampSigned((float) vel.y / ObservationSchema.MAX_SPEED);
        double yawRad = Math.toRadians(self.getYaw());
        double pitchRad = Math.toRadians(self.getPitch());
        obsOut[ObservationSchema.SELF_YAW_SIN] = (float) Math.sin(yawRad);
        obsOut[ObservationSchema.SELF_YAW_COS] = (float) Math.cos(yawRad);
        obsOut[ObservationSchema.SELF_PITCH_SIN] = (float) Math.sin(pitchRad);
        obsOut[ObservationSchema.SELF_PITCH_COS] = (float) Math.cos(pitchRad);
        obsOut[ObservationSchema.SELF_ON_GROUND] = self.isOnGround() ? 1f : 0f;
        obsOut[ObservationSchema.SELF_IS_AIRBORNE] = self.isOnGround() ? 0f : 1f;
        obsOut[ObservationSchema.SELF_IS_SPRINTING] = self.isSprinting() ? 1f : 0f;
        obsOut[ObservationSchema.SELF_IS_SNEAKING] = self.isSneaking() ? 1f : 0f;
        obsOut[ObservationSchema.SELF_IS_JUMPING] = (self.input != null && self.input.jumping) ? 1f : 0f;
        obsOut[ObservationSchema.SELF_FALL_DISTANCE_NORM] = (float) clamp01(self.fallDistance / 20f);
        obsOut[ObservationSchema.SELF_ATTACK_COOLDOWN_PROGRESS] = clamp01(self.getAttackCooldownProgress(0f));
        obsOut[ObservationSchema.SELF_HAS_SWORD] = isSword(self) ? 1f : 0f;
        obsOut[ObservationSchema.SELF_HAS_SHIELD_RAISED] = self.isBlocking() ? 1f : 0f;
        obsOut[ObservationSchema.SELF_HUNGER_FRACTION] = clamp01(self.getHungerManager().getFoodLevel() / 20f);
        obsOut[ObservationSchema.SELF_XP_LEVEL_NORM] = clamp01(self.experienceLevel / 30f);
        obsOut[ObservationSchema.SELF_IS_IN_WATER] = self.isTouchingWater() ? 1f : 0f;
        obsOut[ObservationSchema.SELF_TICKS_EXISTED_PHASE] = (self.age % 20) / 20f;

        if (currentOpponent != null && currentOpponent.isAlive()) {
            obsOut[ObservationSchema.OPP_PRESENT] = 1f;
            Vec3d oppPos = new Vec3d(currentOpponent.getX(), currentOpponent.getY(), currentOpponent.getZ());
            Vec3d selfPos = new Vec3d(self.getX(), self.getY(), self.getZ());
            Vec3d d = oppPos.subtract(selfPos);
            obsOut[ObservationSchema.OPP_REL_X] = clampSigned((float) d.x / ObservationSchema.MAX_RANGE);
            obsOut[ObservationSchema.OPP_REL_Y] = clampSigned((float) d.y / ObservationSchema.MAX_RANGE);
            obsOut[ObservationSchema.OPP_REL_Z] = clampSigned((float) d.z / ObservationSchema.MAX_RANGE);
            Vec3d relVel = currentOpponent.getVelocity().subtract(vel);
            obsOut[ObservationSchema.OPP_REL_VEL_X] = clampSigned((float) relVel.x / ObservationSchema.MAX_SPEED);
            obsOut[ObservationSchema.OPP_REL_VEL_Y] = clampSigned((float) relVel.y / ObservationSchema.MAX_SPEED);
            obsOut[ObservationSchema.OPP_REL_VEL_Z] = clampSigned((float) relVel.z / ObservationSchema.MAX_SPEED);
            double dist = selfPos.distanceTo(oppPos);
            double horizDist = Math.sqrt(d.x * d.x + d.z * d.z);
            obsOut[ObservationSchema.OPP_DISTANCE_NORM] = clamp01((float) (dist / ObservationSchema.MAX_RANGE));
            obsOut[ObservationSchema.OPP_HORIZ_DISTANCE_NORM] = clamp01((float) (horizDist / ObservationSchema.MAX_RANGE));

            double bearing = Math.atan2(-d.x, d.z); 
            obsOut[ObservationSchema.OPP_REL_YAW_SIN] = (float) Math.sin(bearing);
            obsOut[ObservationSchema.OPP_REL_YAW_COS] = (float) Math.cos(bearing);
            double elevation = Math.atan2(-d.y, horizDist);
            obsOut[ObservationSchema.OPP_REL_PITCH_SIN] = (float) Math.sin(elevation);
            obsOut[ObservationSchema.OPP_REL_PITCH_COS] = (float) Math.cos(elevation);

            double yawErr = normalizeAngle(Math.toRadians(self.getYaw()) - bearing);
            obsOut[ObservationSchema.OPP_YAW_ERROR_SIN] = (float) Math.sin(yawErr);
            obsOut[ObservationSchema.OPP_YAW_ERROR_COS] = (float) Math.cos(yawErr);
            double pitchErr = normalizeAngle(Math.toRadians(self.getPitch()) - elevation);
            obsOut[ObservationSchema.OPP_PITCH_ERROR_SIN] = (float) Math.sin(pitchErr);
            obsOut[ObservationSchema.OPP_PITCH_ERROR_COS] = (float) Math.cos(pitchErr);

            obsOut[ObservationSchema.OPP_HEALTH_FRACTION] = clamp01(currentOpponent.getHealth() / Math.max(1f, currentOpponent.getMaxHealth()));
            obsOut[ObservationSchema.OPP_IS_AIRBORNE] = currentOpponent.isOnGround() ? 0f : 1f;
            obsOut[ObservationSchema.OPP_ON_GROUND] = currentOpponent.isOnGround() ? 1f : 0f;
            obsOut[ObservationSchema.OPP_IS_SPRINTING] = currentOpponent.isSprinting() ? 1f : 0f;
            obsOut[ObservationSchema.OPP_IS_SNEAKING] = currentOpponent.isSneaking() ? 1f : 0f;
            obsOut[ObservationSchema.OPP_IS_BLOCKING] = currentOpponent.isBlocking() ? 1f : 0f;

            double oppMoveDir = Math.atan2(-currentOpponent.getVelocity().x, currentOpponent.getVelocity().z);
            obsOut[ObservationSchema.OPP_MOVE_DIR_SIN] = (float) Math.sin(oppMoveDir);
            obsOut[ObservationSchema.OPP_MOVE_DIR_COS] = (float) Math.cos(oppMoveDir);

            obsOut[ObservationSchema.OPP_IN_VISIBLE_CONE] = Math.abs(yawErr) < Math.toRadians(60) ? 1f : 0f;
            obsOut[ObservationSchema.OPP_LINE_OF_SIGHT_CLEAR] = hasLineOfSight(self, currentOpponent) ? 1f : 0f;
            obsOut[ObservationSchema.OPP_IN_ATTACK_RANGE] = dist <= 3.0 ? 1f : 0f;
            obsOut[ObservationSchema.OPP_IS_TARGET_CENTERED] =
                (Math.abs(yawErr) < Math.toRadians(5) && Math.abs(pitchErr) < Math.toRadians(5)) ? 1f : 0f;
        }

        obsOut[ObservationSchema.COMBAT_TIME_SINCE_ATTACK_NORM] = clamp01(ticksSinceOwnAttack / 40f);
        obsOut[ObservationSchema.COMBAT_TIME_SINCE_HIT_LANDED_NORM] = clamp01(ticksSinceHitLanded / 100f);
        obsOut[ObservationSchema.COMBAT_TIME_SINCE_DAMAGE_TAKEN_NORM] = clamp01(ticksSinceDamageTaken / 100f);
        obsOut[ObservationSchema.COMBAT_RECENT_DAMAGE_DEALT_NORM] = clamp01(recentDamageDealt / 20f);
        obsOut[ObservationSchema.COMBAT_RECENT_DAMAGE_RECEIVED_NORM] = clamp01(recentDamageReceived / 20f);
        obsOut[ObservationSchema.COMBAT_LAST_ATTACK_HIT] = lastAttackHit ? 1f : 0f;
        obsOut[ObservationSchema.COMBAT_LAST_ATTACK_MISSED] = lastAttackMissed ? 1f : 0f;
        obsOut[ObservationSchema.COMBAT_IS_IN_COMBAT] = (ticksSinceDamageTaken < 100 || ticksSinceHitLanded < 100) ? 1f : 0f;
        obsOut[ObservationSchema.COMBAT_CRIT_AVAILABLE] = (!self.isOnGround() && vel.y < 0) ? 1f : 0f;

        fillSpatialRaycasts(self, obsOut);
        fillTargetBlock(self, obsOut);

        int base = ObservationSchema.PREV_MOVE_ONEHOT_START;
        if (prevMoveAction >= 0 && prevMoveAction < ActionSpace.MOVE_ACTIONS) obsOut[base + prevMoveAction] = 1f;
        obsOut[ObservationSchema.PREV_JUMP] = prevJump ? 1f : 0f;
        obsOut[ObservationSchema.PREV_SPRINT] = prevSprint ? 1f : 0f;
        obsOut[ObservationSchema.PREV_SNEAK] = prevSneak ? 1f : 0f;
        obsOut[ObservationSchema.PREV_ATTACK] = prevAttack ? 1f : 0f;
        double prevYawRad = Math.toRadians(prevYawDeltaDeg);
        double prevPitchRad = Math.toRadians(prevPitchDeltaDeg);
        obsOut[ObservationSchema.PREV_YAW_ACTION_SIN] = (float) Math.sin(prevYawRad);
        obsOut[ObservationSchema.PREV_YAW_ACTION_COS] = (float) Math.cos(prevYawRad);
        obsOut[ObservationSchema.PREV_PITCH_ACTION_SIN] = (float) Math.sin(prevPitchRad);
        obsOut[ObservationSchema.PREV_PITCH_ACTION_COS] = (float) Math.cos(prevPitchRad);

        obsOut[ObservationSchema.META_EPISODE_TIME_NORM] = clamp01(ticksThisEpisode / 1200f);
        obsOut[ObservationSchema.META_OPPONENT_TIER_LOW] = opponentTier == 0 ? 1f : 0f;
        obsOut[ObservationSchema.META_OPPONENT_TIER_AVERAGE] = opponentTier == 1 ? 1f : 0f;
        obsOut[ObservationSchema.META_OPPONENT_TIER_HIGH] = opponentTier == 2 ? 1f : 0f;
        if (self.getEntityWorld() != null) {
            long t = self.getEntityWorld().getTimeOfDay() % 24000;
            double phase = 2 * Math.PI * t / 24000.0;
            obsOut[ObservationSchema.META_TIME_OF_DAY_SIN] = (float) Math.sin(phase);
            obsOut[ObservationSchema.META_TIME_OF_DAY_COS] = (float) Math.cos(phase);
        }
    }

    private void fillTargetBlock(ClientPlayerEntity self, float[] obsOut) {
        if (currentOpponent == null || !currentOpponent.isAlive()) return;
        Vec3d oppPos = new Vec3d(currentOpponent.getX(), currentOpponent.getY(), currentOpponent.getZ());
        Vec3d d = oppPos.add(0, currentOpponent.getStandingEyeHeight() * 0.5, 0)
                                   .subtract(self.getEyePos());
        double dist = d.length();
        double horiz = Math.sqrt(d.x * d.x + d.z * d.z);
        double bearing = normalizeAngle(Math.atan2(-d.x, d.z) - Math.toRadians(self.getYaw()));
        double elevation = normalizeAngle(Math.atan2(-d.y, horiz) - Math.toRadians(self.getPitch()));

        obsOut[ObservationSchema.TARGET_BEARING_SIN] = (float) Math.sin(bearing);
        obsOut[ObservationSchema.TARGET_BEARING_COS] = (float) Math.cos(bearing);
        obsOut[ObservationSchema.TARGET_ELEVATION_SIN] = (float) Math.sin(elevation);
        obsOut[ObservationSchema.TARGET_ELEVATION_COS] = (float) Math.cos(elevation);
        double totalAngle = Math.toDegrees(Math.acos(clampSigned((float) Math.cos(bearing) * (float) Math.cos(elevation))));
        obsOut[ObservationSchema.TARGET_ANGULAR_ERROR_NORM] = clamp01((float) (totalAngle / 180.0));
        obsOut[ObservationSchema.TARGET_DISTANCE_NORM] = clamp01((float) (dist / ObservationSchema.MAX_RANGE));
        obsOut[ObservationSchema.TARGET_IS_CENTERED] =
            (Math.abs(bearing) < Math.toRadians(5) && Math.abs(elevation) < Math.toRadians(5)) ? 1f : 0f;

        Vec3d oppVel = currentOpponent.getVelocity();
        double closing = -(d.normalize().dotProduct(oppVel.subtract(self.getVelocity())));
        obsOut[ObservationSchema.TARGET_CLOSING_SPEED_NORM] = clampSigned((float) (closing / ObservationSchema.MAX_SPEED));

        Vec3d predicted = oppPos.add(oppVel.multiply(2));
        Vec3d dp = predicted.subtract(self.getEyePos());
        double horizP = Math.sqrt(dp.x * dp.x + dp.z * dp.z);
        double bearingP = normalizeAngle(Math.atan2(-dp.x, dp.z) - Math.toRadians(self.getYaw()));
        double elevationP = normalizeAngle(Math.atan2(-dp.y, horizP) - Math.toRadians(self.getPitch()));
        obsOut[ObservationSchema.TARGET_PREDICTED_BEARING_SIN] = (float) Math.sin(bearingP);
        obsOut[ObservationSchema.TARGET_PREDICTED_BEARING_COS] = (float) Math.cos(bearingP);
        obsOut[ObservationSchema.TARGET_PREDICTED_ELEVATION_SIN] = (float) Math.sin(elevationP);
        obsOut[ObservationSchema.TARGET_PREDICTED_ELEVATION_COS] = (float) Math.cos(elevationP);
    }

    private void fillSpatialRaycasts(ClientPlayerEntity self, float[] obsOut) {
        if (self.getEntityWorld() == null) return;
        Vec3d origin = self.getEyePos();
        double[] verticalOffsets = {-1.2, -0.4, 0.0, 0.6, 1.2}; 

        for (int dir = 0; dir < ObservationSchema.SPATIAL_RAYCAST_DIRS; dir++) {
            double angle = 2 * Math.PI * dir / ObservationSchema.SPATIAL_RAYCAST_DIRS;
            double dx = Math.sin(angle), dz = Math.cos(angle);
            for (int band = 0; band < ObservationSchema.SPATIAL_RAYCAST_BANDS; band++) {
                Vec3d from = origin.add(0, verticalOffsets[band], 0);
                Vec3d to = from.add(dx * ObservationSchema.MAX_RANGE, 0, dz * ObservationSchema.MAX_RANGE);
                float normalizedDist = 1.0f; 
                var ctx = new RaycastContext(
                    from, to,
                    RaycastContext.ShapeType.COLLIDER,
                    RaycastContext.FluidHandling.NONE,
                    self);
                HitResult result = self.getEntityWorld().raycast(ctx);
                if (result != null && result.getType() == HitResult.Type.BLOCK) {
                    double d = result.getPos().distanceTo(from);
                    normalizedDist = clamp01((float) (d / ObservationSchema.MAX_RANGE));
                }
                obsOut[ObservationSchema.spatialIndex(dir, band)] = normalizedDist;
            }
        }
    }

    private boolean hasLineOfSight(ClientPlayerEntity self, LivingEntity target) {
        if (self.getEntityWorld() == null) return false;
        var ctx = new RaycastContext(
            self.getEyePos(), target.getEyePos(),
            RaycastContext.ShapeType.COLLIDER,
            RaycastContext.FluidHandling.NONE,
            self
        );
        HitResult result = self.getEntityWorld().raycast(ctx);
        return result.getType() == HitResult.Type.MISS;
    }

    private boolean isSword(ClientPlayerEntity self) {
        var stack = self.getMainHandStack();
        return stack != null && stack.getItem().toString().toLowerCase().contains("sword");
    }

    private static float clamp01(float v) { return Math.max(0f, Math.min(1f, v)); }
    private static float clampSigned(float v) { return Math.max(-1f, Math.min(1f, v)); }
    private static double normalizeAngle(double rad) {
        while (rad > Math.PI) rad -= 2 * Math.PI;
        while (rad < -Math.PI) rad += 2 * Math.PI;
        return rad;
    }
}
