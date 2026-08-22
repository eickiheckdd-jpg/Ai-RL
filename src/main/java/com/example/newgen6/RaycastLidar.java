package com.example.newgen6.client;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;
import net.minecraft.world.World;

/**
 * Fires BotState.LIDAR_RAYS rays in a cone around the player's look vector
 * and returns normalized distances (0 = touching something, 1 = clear out
 * to maxRange). Used as short-range obstacle/terrain awareness.
 *
 * NOTE: RaycastContext constructor args and World#raycast signature are the
 * long-standing Yarn convention — verify against your 1.21.11 decompiled
 * sources if this doesn't compile as-is.
 */
public class RaycastLidar {

    private static final double MAX_RANGE = 6.0;
    private static final double CONE_SPREAD_DEGREES = 40.0; // total spread across all rays

    public static double[] castCone(World world, ClientPlayerEntity player, int rayCount) {
        double[] distances = new double[rayCount];
        Vec3d eyePos = player.getEyePos();
        Vec3d lookVec = player.getRotationVec(1.0f);

        // Build an orthonormal basis around the look vector so we can offset rays sideways.
        Vec3d worldUp = new Vec3d(0, 1, 0);
        Vec3d right = lookVec.crossProduct(worldUp).normalize();
        Vec3d up = right.crossProduct(lookVec).normalize();

        for (int i = 0; i < rayCount; i++) {
            // Spread rays evenly across the cone, centered on the look vector.
            double t = rayCount == 1 ? 0 : (i / (double) (rayCount - 1)) - 0.5;
            double angleRad = Math.toRadians(t * CONE_SPREAD_DEGREES);

            Vec3d rayDir = lookVec
                    .multiply(Math.cos(angleRad))
                    .add(right.multiply(Math.sin(angleRad)))
                    .normalize();

            Vec3d end = eyePos.add(rayDir.multiply(MAX_RANGE));

            RaycastContext ctx = new RaycastContext(
                    eyePos, end,
                    RaycastContext.ShapeType.COLLIDER,
                    RaycastContext.FluidHandling.NONE,
                    player
            );
            HitResult hit = world.raycast(ctx);

            double dist = MAX_RANGE;
            if (hit != null && hit.getType() != HitResult.Type.MISS) {
                dist = hit.getPos().distanceTo(eyePos);
            }
            distances[i] = clamp01(dist / MAX_RANGE);
        }
        return distances;
    }

    private static double clamp01(double v) {
        return Math.max(0.0, Math.min(1.0, v));
    }
}
