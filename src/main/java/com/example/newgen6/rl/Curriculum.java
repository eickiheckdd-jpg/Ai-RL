package com.example.newgen6.rl;

/**
 * 7-stage curriculum.
 *
 * Stage 0 – Neutral initialized (high entropy, large look scale, random exploration / shaking)
 * Stage 1 – Movement & smooth aiming
 * Stage 2 – Basic spacing
 * Stage 3 – Attack timing & single hits
 * Stage 4 – Combos (W-tap, S-tap, jump-reset)
 * Stage 5 – Self-play / opponent adaptation
 * Stage 6 – HT3 refinement (full reward, low noise, precise look)
 */
public final class Curriculum {

    private int stage = 0;
    private long totalSteps = 0;
    private float entropyCoef = Config.ENTROPY_COEF_START;
    private float noiseScale = Config.INITIAL_NOISE;
    private float lookScale = Config.LOOK_SCALE_START;

    public void step(long globalSteps) {
        totalSteps = globalSteps;
        // advance stage by thresholds
        int newStage = 0;
        for (int s = Config.STAGES - 1; s >= 0; s--) {
            if (globalSteps >= Config.STAGE_THRESHOLDS[s]) {
                newStage = s;
                break;
            }
        }
        if (newStage != stage) {
            stage = newStage;
            onStageChange();
        }
        // continuous annealing inside stage
        float progress = Math.min(1f, totalSteps / 600_000f);
        entropyCoef = lerp(Config.ENTROPY_COEF_START, Config.ENTROPY_COEF_END, progress);
        noiseScale = lerp(Config.INITIAL_NOISE, Config.FINAL_NOISE, progress);
        lookScale = lerp(Config.LOOK_SCALE_START, Config.LOOK_SCALE_END, progress);
    }

    private void onStageChange() {
        // can log or reset certain stats here
    }

    public int getStage() { return stage; }
    public float getEntropyCoef() { return entropyCoef; }
    public float getNoiseScale() { return noiseScale; }
    public float getLookScale() { return lookScale; }
    public long getTotalSteps() { return totalSteps; }

    public String stageName() {
        return switch (stage) {
            case 0 -> "0-Neutral (shaking exploration)";
            case 1 -> "1-Movement & Aiming";
            case 2 -> "2-Spacing";
            case 3 -> "3-Attack Timing";
            case 4 -> "4-Combos";
            case 5 -> "5-Self-Play";
            case 6 -> "6-HT3 Refinement";
            default -> "Unknown";
        };
    }

    private static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
}
