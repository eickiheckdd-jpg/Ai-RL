package com.example.newgen6;

import java.util.Arrays;

/**
 * NewGen6 feature-history/tensor builder.
 *
 * This class contains ONLY the feature schema and tensor mechanics that are
 * supported by the supplied artifacts. It deliberately does NOT invent the
 * missing Minecraft raw-feature formulas.
 *
 * Raw input:
 *   float[78]
 *
 * Per-frame packed input:
 *   0..77   = normalized current raw feature
 *   78..155 = normalized current - normalized previous
 *
 * Final model input:
 *   float[1][96][156]
 */
public final class FeatureTensor {

    public static final int RAW_FEATURES = 78;
    public static final int INPUT_FEATURES = 156;
    public static final int HISTORY_LENGTH = 96;

    public static final String[] FEATURE_NAMES = {
            "dist",
            "rel_dx",
            "rel_dy",
            "rel_dz",
            "d_dist",
            "d_rel_dx",
            "d_rel_dz",
            "dt_ms",
            "s_yaw_sin",
            "s_yaw_cos",
            "s_pitch",
            "o_yaw_sin",
            "o_yaw_cos",
            "o_pitch",
            "s_vx",
            "s_vy",
            "s_vz",
            "s_speed",
            "s_ground",
            "s_sprint",
            "s_fall",
            "s_hurt",
            "s_hp",
            "s_weapon",
            "s_using",
            "s_sneak",
            "s_attack_strength_pre",
            "ms_since_attack_pre",
            "self_attack_recent_pre",
            "o_ground",
            "o_sprint",
            "o_fall",
            "o_block",
            "o_hurt",
            "o_hp",
            "o_using",
            "o_sneak",
            "o_vx",
            "o_vy",
            "o_vz",
            "o_speed",
            "o_attack_strength",
            "o_swing",
            "o_swing_t",
            "align_self",
            "align_opp",
            "m_aggression",
            "m_spacing",
            "m_interval_ms",
            "m_hit_range",
            "aim_yaw_err",
            "aim_pitch_err",
            "aim_err",
            "on_target",
            "box_dist",
            "eye_dist",
            "opp_aim_yaw_err",
            "opp_aim_pitch_err",
            "opp_aim_err",
            "opp_on_us",
            "opp_box_dist",
            "opp_eye_dist",
            "aim_disadvantage",
            "s_ping",
            "o_ping",
            "ping_gap",
            "ms_since_opp_swing",
            "opp_swing_recent",
            "self_side_speed",
            "self_closing_speed",
            "o_side_speed",
            "o_closing_speed",
            "o_side_abs",
            "schema_31",
            "schema_32",
            "schema_42",
            "schema_51",
            "schema_58"
    };

    private final float[] mean;
    private final float[] std;

    /*
     * Stored oldest -> newest.
     *
     * Each packed frame is:
     *   [0..77]   normalized current
     *   [78..155] normalized current - normalized previous
     */
    private final float[][] history =
            new float[HISTORY_LENGTH][INPUT_FEATURES];

    private int size;

    private FeatureTensor(float[] mean, float[] std) {
        validateStats(mean, std);
        this.mean = mean.clone();
        this.std = std.clone();
    }

    /**
     * Creates a builder using the exact per-feature mean/std values from the
     * supplied feature specification.
     */
    public static FeatureTensor create(
            float[] mean,
            float[] std
    ) {
        return new FeatureTensor(mean, std);
    }

    /**
     * Adds one raw 78-feature frame.
     *
     * IMPORTANT:
     * The caller must provide the real, correctly calculated Minecraft
     * features. This class does not invent those formulas.
     */
    public synchronized void pushRawFrame(float[] raw) {
        validateRawFrame(raw);

        float[] normalized = new float[RAW_FEATURES];

        for (int i = 0; i < RAW_FEATURES; i++) {
            normalized[i] =
                    (raw[i] - mean[i]) / std[i];

            if (!Float.isFinite(normalized[i])) {
                throw new IllegalArgumentException(
                        "NewGen6: normalized feature " +
                        i +
                        " (" +
                        FEATURE_NAMES[i] +
                        ") is not finite."
                );
            }
        }

        float[] packed =
                new float[INPUT_FEATURES];

        System.arraycopy(
                normalized,
                0,
                packed,
                0,
                RAW_FEATURES
        );

        /*
         * The supplied preprocessing contract uses current-current for the
         * first delta, producing an all-zero delta block for the first frame.
         */
        if (size == 0) {
            Arrays.fill(
                    packed,
                    RAW_FEATURES,
                    INPUT_FEATURES,
                    0.0f
            );
        } else {
            float[] previous =
                    history[size - 1];

            for (int i = 0; i < RAW_FEATURES; i++) {
                packed[RAW_FEATURES + i] =
                        normalized[i] - previous[i];
            }
        }

        appendPackedFrame(packed);
    }

    private void appendPackedFrame(float[] packed) {

        if (size < HISTORY_LENGTH) {

            System.arraycopy(
                    packed,
                    0,
                    history[size],
                    0,
                    INPUT_FEATURES
            );

            size++;
            return;
        }

        /*
         * Roll the oldest frame out.
         */
        for (int frame = 1;
             frame < HISTORY_LENGTH;
             frame++) {

            System.arraycopy(
                    history[frame],
                    0,
                    history[frame - 1],
                    0,
                    INPUT_FEATURES
            );
        }

        System.arraycopy(
                packed,
                0,
                history[HISTORY_LENGTH - 1],
                0,
                INPUT_FEATURES
        );
    }

    /**
     * Produces the exact shape expected by ModelRunner:
     * [1][96][156].
     *
     * Before 96 frames are available, unused leading history entries remain
     * zero. This warm-up behavior should be replaced if the original training
     * source later proves a different policy.
     */
    public synchronized float[][][] toModelInput() {

        float[][][] tensor =
                new float[1][HISTORY_LENGTH][INPUT_FEATURES];

        for (int frame = 0;
             frame < HISTORY_LENGTH;
             frame++) {

            System.arraycopy(
                    history[frame],
                    0,
                    tensor[0][frame],
                    0,
                    INPUT_FEATURES
            );
        }

        return tensor;
    }

    /**
     * Returns the newest packed 156-value frame.
     */
    public synchronized float[] latestFrame() {

        float[] result =
                new float[INPUT_FEATURES];

        if (size == 0) {
            return result;
        }

        System.arraycopy(
                history[size - 1],
                0,
                result,
                0,
                INPUT_FEATURES
        );

        return result;
    }

    public synchronized int frameCount() {
        return size;
    }

    public synchronized boolean isWarm() {
        return size >= HISTORY_LENGTH;
    }

    public synchronized void clear() {

        for (float[] frame : history) {
            Arrays.fill(frame, 0.0f);
        }

        size = 0;
    }

    /**
     * Calls the real inference method already present in ModelRunner.
     */
    public synchronized ModelRunner.InferenceResult infer() {

        if (!isWarm()) {
            throw new IllegalStateException(
                    "NewGen6: Need " +
                    HISTORY_LENGTH +
                    " frames before inference; have " +
                    size
            );
        }

        return ModelRunner.runInference(
                toModelInput()
        );
    }

    /**
     * Prints a compact sample for live debugging without dumping all
     * 14,976 tensor values.
     */
    public synchronized void logLatestFrameSample() {

        if (size == 0) {
            System.out.println(
                    "NewGen6: FeatureTensor has no frames."
            );
            return;
        }

        float[] frame =
                history[size - 1];

        System.out.println(
                "NewGen6: FeatureTensor frame=" +
                (size - 1) +
                " rawFeatures=" +
                RAW_FEATURES +
                " packedFeatures=" +
                INPUT_FEATURES +
                " history=" +
                size +
                "/" +
                HISTORY_LENGTH
        );

        /*
         * These are only debug samples. The entire schema is retained in
         * FEATURE_NAMES above.
         */
        int[] sampleIndices = {
                0, 1, 2, 3, 7,
                8, 10, 14, 17, 18,
                21, 22, 26, 29, 34,
                37, 40, 44, 45,
                46, 47, 48, 49,
                50, 52, 53, 63,
                65, 68, 69, 70, 71,
                72, 73, 74, 75, 76, 77
        };

        for (int index : sampleIndices) {

            System.out.println(
                    "NewGen6: " +
                    index +
                    " " +
                    FEATURE_NAMES[index] +
                    " normalized=" +
                    frame[index] +
                    " delta=" +
                    frame[RAW_FEATURES + index]
            );
        }
    }

    private static void validateRawFrame(float[] raw) {

        if (raw == null) {
            throw new IllegalArgumentException(
                    "NewGen6: raw feature frame is null."
            );
        }

        if (raw.length != RAW_FEATURES) {
            throw new IllegalArgumentException(
                    "NewGen6: expected " +
                    RAW_FEATURES +
                    " raw features, got " +
                    raw.length
            );
        }

        for (int i = 0; i < RAW_FEATURES; i++) {

            if (!Float.isFinite(raw[i])) {

                throw new IllegalArgumentException(
                        "NewGen6: raw feature " +
                        i +
                        " (" +
                        FEATURE_NAMES[i] +
                        ") is not finite: " +
                        raw[i]
                );
            }
        }
    }

    private static void validateStats(
            float[] mean,
            float[] std
    ) {

        if (mean == null || std == null) {
            throw new IllegalArgumentException(
                    "NewGen6: mean/std cannot be null."
            );
        }

        if (mean.length != RAW_FEATURES) {
            throw new IllegalArgumentException(
                    "NewGen6: expected " +
                    RAW_FEATURES +
                    " means, got " +
                    mean.length
            );
        }

        if (std.length != RAW_FEATURES) {
            throw new IllegalArgumentException(
                    "NewGen6: expected " +
                    RAW_FEATURES +
                    " std values, got " +
                    std.length
            );
        }

        for (int i = 0; i < RAW_FEATURES; i++) {

            if (!Float.isFinite(mean[i]) ||
                !Float.isFinite(std[i]) ||
                std[i] <= 0.0f) {

                throw new IllegalArgumentException(
                        "NewGen6: invalid normalization statistics at " +
                        i +
                        " (" +
                        FEATURE_NAMES[i] +
                        "): mean=" +
                        mean[i] +
                        ", std=" +
                        std[i]
                );
            }
        }
    }
}