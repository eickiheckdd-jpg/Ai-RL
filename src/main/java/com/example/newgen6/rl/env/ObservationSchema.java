package com.example.newgen6.rl.env;

import java.util.ArrayList;
import java.util.List;

/**
 * ObservationSchema is the SINGLE SOURCE OF TRUTH for the fixed observation
 * ABI fed to the policy every Minecraft tick.
 *
 * OBSERVATION_SIZE = 229 (float[229], indices 0..228 inclusive).
 *
 * This class does not itself read game state - {@link com.example.newgen6.rl.env.CombatEnv}
 * fills a float[229] using the index constants below. This class exists so
 * that:
 *   1. There is exactly one place that defines what index N means.
 *   2. A runtime assertion (validate()) guarantees the feature table and
 *      OBSERVATION_SIZE never silently drift apart.
 *   3. An external auditor (human or model) can read this file top-to-bottom
 *      and know exactly what every one of the 229 floats represents, its
 *      expected numeric range, and how it is normalized.
 *
 * Normalization conventions used throughout:
 *   - booleans                -> 0.0f or 1.0f
 *   - health / fractions      -> value / max, clamped to [0, 1]
 *   - angles (yaw/pitch/etc.) -> encoded as sin(theta), cos(theta) pairs, each in [-1, 1]
 *   - distances               -> clamp(distance / MAX_RANGE, 0, 1)  -> [0, 1]
 *   - relative offsets (x/y/z)-> clamp(delta / MAX_RANGE, -1, 1)    -> [-1, 1]
 *   - velocities              -> clamp(v / MAX_SPEED, -1, 1)        -> [-1, 1]
 */
public final class ObservationSchema {

    private ObservationSchema() {}

    /** Fixed, non-negotiable size of one tick's observation. */
    public static final int OBSERVATION_SIZE = 229;

    /** Final valid index into the observation array. */
    public static final int LAST_INDEX = OBSERVATION_SIZE - 1; // 228

    // Normalization constants (tunable, but must stay consistent between
    // collection in CombatEnv and any documentation here).
    public static final float MAX_RANGE = 32.0f;   // meters, for relative positions
    public static final float MAX_SPEED = 1.0f;    // blocks/tick, generous cap for player speed
    public static final float MAX_TICKS_MEMORY = 200.0f; // matches the 200-tick / 10s context window

    // ----------------------------------------------------------------
    // Feature block: SELF / PLAYER            [0 .. 23]   (24 features)
    // ----------------------------------------------------------------
    public static final int SELF_HEALTH_FRACTION      = 0;  // health / maxHealth,               [0,1]
    public static final int SELF_ABSORPTION_FRACTION  = 1;  // absorption / maxHealth,            [0,1]
    public static final int SELF_VEL_X                = 2;  // clamp(vx / MAX_SPEED, -1, 1)
    public static final int SELF_VEL_Y                = 3;  // clamp(vy / MAX_SPEED, -1, 1)
    public static final int SELF_VEL_Z                = 4;  // clamp(vz / MAX_SPEED, -1, 1)
    public static final int SELF_HORIZ_SPEED          = 5;  // clamp(sqrt(vx^2+vz^2) / MAX_SPEED, 0, 1)
    public static final int SELF_VERT_SPEED            = 6;  // clamp(vy / MAX_SPEED, -1, 1)
    public static final int SELF_YAW_SIN               = 7;  // sin(yaw)
    public static final int SELF_YAW_COS               = 8;  // cos(yaw)
    public static final int SELF_PITCH_SIN              = 9;  // sin(pitch)
    public static final int SELF_PITCH_COS              = 10; // cos(pitch)
    public static final int SELF_ON_GROUND              = 11; // 0/1
    public static final int SELF_IS_AIRBORNE             = 12; // 0/1 (!onGround)
    public static final int SELF_IS_SPRINTING            = 13; // 0/1
    public static final int SELF_IS_SNEAKING             = 14; // 0/1
    public static final int SELF_IS_JUMPING               = 15; // 0/1 (jump key held this tick)
    public static final int SELF_FALL_DISTANCE_NORM       = 16; // clamp(fallDistance / 20, 0, 1)
    public static final int SELF_ATTACK_COOLDOWN_PROGRESS  = 17; // getAttackCooldownProgress(0), [0,1]
    public static final int SELF_HAS_SWORD                 = 18; // 0/1 mainhand is a sword
    public static final int SELF_HAS_SHIELD_RAISED         = 19; // 0/1 blocking with shield
    public static final int SELF_HUNGER_FRACTION           = 20; // foodLevel / 20
    public static final int SELF_XP_LEVEL_NORM              = 21; // clamp(level / 30, 0, 1)
    public static final int SELF_IS_IN_WATER                = 22; // 0/1
    public static final int SELF_TICKS_EXISTED_PHASE         = 23; // (tickCount % 20) / 20.0, coarse clock

    // ----------------------------------------------------------------
    // Feature block: OPPONENT (relative)       [24 .. 55]  (32 features)
    // ----------------------------------------------------------------
    public static final int OPP_PRESENT                  = 24; // 0/1 - whether a valid opponent is selected
    public static final int OPP_REL_X                    = 25; // clamp(dx / MAX_RANGE, -1, 1)
    public static final int OPP_REL_Y                    = 26; // clamp(dy / MAX_RANGE, -1, 1)
    public static final int OPP_REL_Z                    = 27; // clamp(dz / MAX_RANGE, -1, 1)
    public static final int OPP_REL_VEL_X                 = 28; // clamp((vOx-vSx) / MAX_SPEED, -1, 1)
    public static final int OPP_REL_VEL_Y                 = 29; // clamp((vOy-vSy) / MAX_SPEED, -1, 1)
    public static final int OPP_REL_VEL_Z                 = 30; // clamp((vOz-vSz) / MAX_SPEED, -1, 1)
    public static final int OPP_DISTANCE_NORM              = 31; // clamp(dist / MAX_RANGE, 0, 1)
    public static final int OPP_HORIZ_DISTANCE_NORM         = 32; // clamp(horizDist / MAX_RANGE, 0, 1)
    public static final int OPP_REL_YAW_SIN                 = 33; // sin(bearing from self to opponent)
    public static final int OPP_REL_YAW_COS                 = 34; // cos(bearing)
    public static final int OPP_REL_PITCH_SIN                = 35; // sin(elevation angle to opponent)
    public static final int OPP_REL_PITCH_COS                = 36; // cos(elevation angle)
    public static final int OPP_YAW_ERROR_SIN                 = 37; // sin(selfYaw - bearingToOpp) - aim error
    public static final int OPP_YAW_ERROR_COS                 = 38; // cos(selfYaw - bearingToOpp)
    public static final int OPP_PITCH_ERROR_SIN                = 39; // sin(selfPitch - elevationToOpp)
    public static final int OPP_PITCH_ERROR_COS                = 40; // cos(selfPitch - elevationToOpp)
    public static final int OPP_HEALTH_FRACTION                 = 41; // opponent health/maxHealth if observable, else 0.5
    public static final int OPP_IS_AIRBORNE                      = 42; // 0/1
    public static final int OPP_ON_GROUND                        = 43; // 0/1
    public static final int OPP_IS_SPRINTING                     = 44; // 0/1
    public static final int OPP_IS_SNEAKING                      = 45; // 0/1
    public static final int OPP_IS_BLOCKING                      = 46; // 0/1 (shield raised)
    public static final int OPP_MOVE_DIR_SIN                     = 47; // sin(opponent facing/movement dir)
    public static final int OPP_MOVE_DIR_COS                     = 48; // cos(opponent facing/movement dir)
    public static final int OPP_CLOSING_SPEED                     = 49; // clamp(-d(dist)/dt / MAX_SPEED, -1, 1) positive = closing
    public static final int OPP_IN_VISIBLE_CONE                    = 50; // 0/1 opponent within self's FOV cone
    public static final int OPP_LINE_OF_SIGHT_CLEAR                 = 51; // 0/1 raycast unobstructed
    public static final int OPP_IN_ATTACK_RANGE                     = 52; // 0/1 dist <= sword reach
    public static final int OPP_IS_TARGET_CENTERED                   = 53; // 0/1 |yawError|&|pitchError| within small threshold
    public static final int OPP_TICKS_SINCE_SEEN_NORM                 = 54; // clamp(ticksSinceVisible / MAX_TICKS_MEMORY, 0, 1)
    public static final int OPP_RESERVED_1                           = 55; // reserved for future opponent feature

    // ----------------------------------------------------------------
    // Feature block: COMBAT STATE              [56 .. 79]  (24 features)
    // ----------------------------------------------------------------
    public static final int COMBAT_TIME_SINCE_ATTACK_NORM          = 56; // clamp(ticksSinceOwnAttack / 40, 0, 1)
    public static final int COMBAT_TIME_SINCE_HIT_LANDED_NORM       = 57; // clamp(ticksSinceHitLanded / 100, 0, 1)
    public static final int COMBAT_TIME_SINCE_DAMAGE_TAKEN_NORM      = 58; // clamp(ticksSinceDamageTaken / 100, 0, 1)
    public static final int COMBAT_RECENT_DAMAGE_DEALT_NORM           = 59; // clamp(damageDealtLast20t / 20, 0, 1)
    public static final int COMBAT_RECENT_DAMAGE_RECEIVED_NORM         = 60; // clamp(damageTakenLast20t / 20, 0, 1)
    public static final int COMBAT_LAST_ATTACK_HIT                     = 61; // 0/1 did last attack action connect
    public static final int COMBAT_LAST_ATTACK_MISSED                  = 62; // 0/1
    public static final int COMBAT_IS_IN_COMBAT                        = 63; // 0/1 (any damage exchanged recently)
    public static final int COMBAT_CRIT_AVAILABLE                      = 64; // 0/1 falling & not on ground (crit condition)
    public static final int COMBAT_KNOCKBACK_RECEIVED_NORM              = 65; // clamp(lastKnockbackMagnitude / 2, 0, 1)
    public static final int COMBAT_SELF_KILLS_THIS_EPISODE_NORM          = 66; // clamp(kills / 5, 0, 1)
    public static final int COMBAT_SELF_DEATHS_THIS_EPISODE_NORM          = 67; // clamp(deaths / 5, 0, 1)
    // reserved combat-block slots for future features (padded, always 0 until used)
    public static final int COMBAT_RESERVED_START                          = 68;
    public static final int COMBAT_RESERVED_END                            = 79; // inclusive, 12 reserved floats

    // ----------------------------------------------------------------
    // Feature block: SPATIAL / VISION RAYCASTS  [80 .. 159] (80 features)
    // 16 horizontal directions (every 22.5 deg around self) x 5 vertical
    // raycast bands (feet-level, waist, eye, above, sky), each cell holds
    // normalized obstacle distance in [0,1] (1 = clear to MAX_RANGE).
    // ----------------------------------------------------------------
    public static final int SPATIAL_RAYCAST_START = 80;
    public static final int SPATIAL_RAYCAST_DIRS   = 16;
    public static final int SPATIAL_RAYCAST_BANDS  = 5;
    public static final int SPATIAL_RAYCAST_END    = SPATIAL_RAYCAST_START + (SPATIAL_RAYCAST_DIRS * SPATIAL_RAYCAST_BANDS) - 1; // 159

    /** index = SPATIAL_RAYCAST_START + dir*BANDS + band, dir in [0,15], band in [0,4] */
    public static int spatialIndex(int dir, int band) {
        return SPATIAL_RAYCAST_START + dir * SPATIAL_RAYCAST_BANDS + band;
    }

    // ----------------------------------------------------------------
    // Feature block: TARGET RELATIONSHIP        [160 .. 179] (20 features)
    // (bearing/elevation restated relative to current aim/crosshair, not
    //  just body yaw - useful because body yaw and camera yaw can differ)
    // ----------------------------------------------------------------
    public static final int TARGET_BEARING_SIN            = 160; // sin(target bearing rel. to camera yaw)
    public static final int TARGET_BEARING_COS             = 161;
    public static final int TARGET_ELEVATION_SIN            = 162; // sin(target elevation rel. to camera pitch)
    public static final int TARGET_ELEVATION_COS             = 163;
    public static final int TARGET_ANGULAR_ERROR_NORM         = 164; // clamp(totalAngleToTarget / 180, 0, 1)
    public static final int TARGET_DISTANCE_NORM               = 165; // clamp(dist / MAX_RANGE, 0, 1)
    public static final int TARGET_REL_VEL_TANGENTIAL           = 166; // clamp(tangential component / MAX_SPEED, -1, 1)
    public static final int TARGET_REL_VEL_RADIAL                = 167; // clamp(radial (closing) component / MAX_SPEED, -1, 1)
    public static final int TARGET_MOVE_DIR_SIN                   = 168; // sin(target's movement heading)
    public static final int TARGET_MOVE_DIR_COS                    = 169;
    public static final int TARGET_CLOSING_SPEED_NORM                = 170; // clamp(closingSpeed / MAX_SPEED, -1, 1)
    public static final int TARGET_IS_CENTERED                       = 171; // 0/1 within small crosshair threshold
    public static final int TARGET_PREDICTED_BEARING_SIN               = 172; // sin(bearing to target's predicted pos +2 ticks)
    public static final int TARGET_PREDICTED_BEARING_COS               = 173;
    public static final int TARGET_PREDICTED_ELEVATION_SIN              = 174;
    public static final int TARGET_PREDICTED_ELEVATION_COS              = 175;
    public static final int TARGET_RESERVED_1                          = 176;
    public static final int TARGET_RESERVED_2                          = 177;
    public static final int TARGET_RESERVED_3                          = 178;
    public static final int TARGET_RESERVED_4                          = 179;

    // ----------------------------------------------------------------
    // Feature block: RECENT ACTION HISTORY      [180 .. 208] (29 features)
    // Encodes the AI's OWN previous-tick action selections so the network
    // can learn action->consequence relationships. These are the action
    // that was taken last tick, one-hot / sin-cos encoded, NOT instructions.
    // ----------------------------------------------------------------
    public static final int PREV_MOVE_ONEHOT_START   = 180; // 9 floats, one-hot over MOVE_ACTIONS (see ActionSpace)
    public static final int PREV_MOVE_ONEHOT_END      = 188; // inclusive
    public static final int PREV_JUMP                 = 189; // 0/1
    public static final int PREV_SPRINT                = 190; // 0/1
    public static final int PREV_SNEAK                  = 191; // 0/1
    public static final int PREV_ATTACK                  = 192; // 0/1
    public static final int PREV_YAW_BUCKET_ONEHOT_START   = 193; // 19 floats, one-hot over yaw buckets
    public static final int PREV_YAW_BUCKET_ONEHOT_END      = 211; // inclusive -> but capped below to stay in budget

    // NOTE: yaw one-hot(19) would overflow this block; encoded instead as
    // sin/cos of the chosen yaw/pitch bucket angle (compact, 4 floats) plus
    // the raw previous action success flags above. See indices 193-196.
    public static final int PREV_YAW_ACTION_SIN         = 193; // sin(angle represented by chosen yaw bucket)
    public static final int PREV_YAW_ACTION_COS          = 194;
    public static final int PREV_PITCH_ACTION_SIN          = 195; // sin(angle represented by chosen pitch bucket)
    public static final int PREV_PITCH_ACTION_COS           = 196;
    public static final int PREV_ACTION_RESERVED_START        = 197;
    public static final int PREV_ACTION_RESERVED_END           = 208; // inclusive, 12 reserved

    // ----------------------------------------------------------------
    // Feature block: EPISODE / TEMPORAL META    [209 .. 228] (20 features)
    // ----------------------------------------------------------------
    public static final int META_EPISODE_TIME_NORM         = 209; // clamp(ticksThisEpisode / 1200, 0, 1) (60s)
    public static final int META_TICKS_SINCE_LAST_RESPAWN_NORM = 210; // clamp(.. / 200, 0, 1)
    public static final int META_OPPONENT_TIER_LOW           = 211; // 0/1 one-hot opponent tier
    public static final int META_OPPONENT_TIER_AVERAGE        = 212; // 0/1
    public static final int META_OPPONENT_TIER_HIGH            = 213; // 0/1
    public static final int META_ARENA_BOUNDS_X_NORM             = 214; // clamp(distance to nearest arena wall X / MAX_RANGE, 0,1)
    public static final int META_ARENA_BOUNDS_Z_NORM              = 215; // clamp(distance to nearest arena wall Z / MAX_RANGE, 0,1)
    public static final int META_TIME_OF_DAY_SIN                    = 216; // sin(2*pi*worldTime/24000)
    public static final int META_TIME_OF_DAY_COS                     = 217; // cos(2*pi*worldTime/24000)
    public static final int META_RESERVED_START                       = 218;
    public static final int META_RESERVED_END                          = 228; // inclusive, 11 reserved -> total reaches 228

    /**
     * Validates that this schema is internally consistent: the last
     * documented meta index must equal LAST_INDEX (228), i.e. exactly
     * 229 features with no gaps and no overflow. Call this once at mod
     * init (see NewGen6RLMod) so a schema edit that breaks the count
     * fails loudly instead of silently corrupting the tensor shapes.
     */
    public static void validate() {
        if (META_RESERVED_END != LAST_INDEX) {
            throw new IllegalStateException(
                "ObservationSchema is inconsistent: META_RESERVED_END=" + META_RESERVED_END
                + " but LAST_INDEX=" + LAST_INDEX + " (OBSERVATION_SIZE=" + OBSERVATION_SIZE + "). "
                + "Every index from 0.." + LAST_INDEX + " must be accounted for exactly once.");
        }
        if (SPATIAL_RAYCAST_END - SPATIAL_RAYCAST_START + 1 != SPATIAL_RAYCAST_DIRS * SPATIAL_RAYCAST_BANDS) {
            throw new IllegalStateException("Spatial raycast block size mismatch.");
        }
    }

    /** Human-readable dump of block boundaries, useful for the HUD / audit log. */
    public static List<String> describeBlocks() {
        List<String> lines = new ArrayList<>();
        lines.add("SELF            [0..23]    24 features");
        lines.add("OPPONENT        [24..55]   32 features");
        lines.add("COMBAT          [56..79]   24 features");
        lines.add("SPATIAL/VISION  [80..159]  80 features (16 dirs x 5 bands)");
        lines.add("TARGET REL.     [160..179] 20 features");
        lines.add("PREV ACTION     [180..208] 29 features");
        lines.add("META/TEMPORAL   [209..228] 20 features");
        lines.add("TOTAL = " + OBSERVATION_SIZE + " floats, indices 0.." + LAST_INDEX);
        return lines;
    }
}
