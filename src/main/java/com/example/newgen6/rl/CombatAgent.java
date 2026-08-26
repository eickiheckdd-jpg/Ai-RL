package com.example.newgen6.rl;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.util.math.MathHelper;

/**
 * Client-side agent loop: observe → act → reward → store → PPO when full.
 * AI starts OFF. When disabled, does not touch inputs.
 *
 * Phase 1 (default): aimOnly = true → movement/attack forced off so PPO only
 * learns yaw/pitch buckets. Flip aimOnly off when expanding to movement.
 */
public final class CombatAgent {
    public final ObservationEncoder encoder = new ObservationEncoder();
    public final ContextBuffer context = new ContextBuffer();
    public final PolicyNetwork policy = new PolicyNetwork(42L);
    public final RolloutBuffer rollout = new RolloutBuffer(RLConstants.PPO_ROLLOUT);
    public final PPOTrainer trainer = new PPOTrainer(policy);

    public boolean aiEnabled;
    public boolean trainEnabled = true;
    public boolean hudEnabled = true;
    /** When true: force HOLD + no jump/sprint/sneak/attack (aim curriculum). */
    public boolean aimOnly = true;

    public long envSteps;
    public float lastReward;
    public float episodeReward;
    public float meanReward;
    public float lastAimError = 1f;
    public ActionSample lastAction;
    public boolean brainLoaded;
    public String lastBrainMessage = "no brain";
    private int savesSinceStart;

    private float prevAimError = 1f;
    private boolean prevHadTarget;
    private final ContextBuffer trainScratch = new ContextBuffer();

    private boolean wantForward, wantBack, wantLeft, wantRight;
    private boolean wantJump, wantSprint, wantSneak, wantAttack;

    public void toggleAi() {
        aiEnabled = !aiEnabled;
        if (!aiEnabled) {
            releaseControls();
            saveBrain(); // checkpoint when disabling AI
        }
    }

    public void toggleHud() {
        hudEnabled = !hudEnabled;
    }

    public void toggleAimOnly() {
        aimOnly = !aimOnly;
    }

    public void releaseControls() {
        wantForward = wantBack = wantLeft = wantRight = false;
        wantJump = wantSprint = wantSneak = wantAttack = false;
    }

    public void clientTick(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null) {
            context.clear();
            return;
        }
        if (!player.isAlive()) {
            encoder.resetEpisode();
            context.clear();
            episodeReward = 0f;
            prevAimError = 1f;
            prevHadTarget = false;
            releaseControls();
            return;
        }

        float[] obs = encoder.encode(client);
        context.push(obs);

        if (!aiEnabled) {
            releaseControls();
            return;
        }

        ActionSample action = policy.act(context, false);
        if (aimOnly) {
            action = ActionSample.aimOnly(action);
        }
        lastAction = action;
        applyAction(client, player, action);

        // Post-action aim error (camera already updated) — credit this tick's aim.
        float aimErr = encoder.hasTarget()
                ? encoder.aimErrorAfterCamera(player)
                : 1f;
        float reward = computeAimReward(aimErr, encoder.hasTarget(), encoder.sameTargetAsPrevious(), player.getPitch());

        prevAimError = encoder.hasTarget() ? aimErr : 1f;
        prevHadTarget = encoder.hasTarget();
        lastAimError = aimErr;
        lastReward = reward;
        episodeReward += reward;
        meanReward = meanReward * 0.99f + reward * 0.01f;

        encoder.onAction(action, reward, aimErr);
        envSteps++;

        if (trainEnabled) {
            // Never start a new rollout while a PPO update is still draining across ticks.
            if (!trainer.isTraining()) {
                boolean done = !player.isAlive();
                rollout.add(obs, context, action, reward, done);
                if (rollout.isFull()) {
                    trainer.beginUpdate(rollout, trainScratch);
                }
            }
        }
        // Spread PPO across ticks — prevents multi-second freezes → 0 FPS.
        if (trainer.isTraining()) {
            boolean finished = trainer.tickWork();
            if (finished) {
                maybeAutosaveBrain();
            }
        }
    }

    /** Load brain from default path if present. Call once on client init. */
    public boolean loadBrain() {
        try {
            java.nio.file.Path path = BrainIO.defaultPath();
            if (!java.nio.file.Files.isRegularFile(path)) {
                lastBrainMessage = "no file";
                brainLoaded = false;
                return false;
            }
            policy.load(path);
            brainLoaded = true;
            lastBrainMessage = "loaded " + path.getFileName();
            return true;
        } catch (Exception e) {
            brainLoaded = false;
            lastBrainMessage = "load fail: " + e.getMessage();
            return false;
        }
    }

    public boolean saveBrain() {
        try {
            java.nio.file.Path path = BrainIO.defaultPath();
            policy.save(path);
            savesSinceStart++;
            lastBrainMessage = "saved #" + savesSinceStart;
            return true;
        } catch (Exception e) {
            lastBrainMessage = "save fail: " + e.getMessage();
            return false;
        }
    }

    private void maybeAutosaveBrain() {
        // After every PPO update
        saveBrain();
    }

    /**
     * Hardened aim reward (phase 1):
     *   absolute term  → pay for low error every tick (stops oscillation farming)
     *   delta term     → only if same target continues (stops switch / spawn spikes)
     *   no target      → mild constant penalty
     * Clipped to [-1, 1].
     */
    private float computeAimReward(float aimErr, boolean hasTarget, boolean sameTarget, float pitchDeg) {
        if (!hasTarget) {
            prevAimError = 1f;
            // Still punish sky-stare with no target
            float sky = Math.max(0f, Math.abs(pitchDeg) - 30f) / 60f;
            return RLConstants.REWARD_NO_TARGET - RLConstants.REWARD_PITCH_EXTREME * sky;
        }
        float absTerm = -RLConstants.REWARD_ABS_WEIGHT * aimErr;
        float deltaTerm = 0f;
        if (sameTarget && prevHadTarget) {
            deltaTerm = RLConstants.REWARD_DELTA_WEIGHT * (prevAimError - aimErr);
        } else {
            deltaTerm = 0f;
            absTerm += RLConstants.REWARD_TARGET_SWITCH;
        }
        // Extreme pitch (sky/ground) is almost never useful in melee — stop drift.
        float sky = Math.max(0f, Math.abs(pitchDeg) - 25f) / 65f;
        float pitchPen = -RLConstants.REWARD_PITCH_EXTREME * sky;
        return MathUtil.clamp(absTerm + deltaTerm + pitchPen, -1f, 1f);
    }

    private void applyAction(MinecraftClient client, ClientPlayerEntity player, ActionSample a) {
        float newYaw = player.getYaw() + a.yawDeltaDeg;
        // Soft combat clamp first (stops random walk to ±90 sky/ground), hard clamp second.
        float newPitch = player.getPitch() + a.pitchDeltaDeg;
        newPitch = MathHelper.clamp(newPitch, -RLConstants.COMBAT_PITCH_MAX, RLConstants.COMBAT_PITCH_MAX);
        newPitch = MathHelper.clamp(newPitch, -90f, 90f);
        player.setYaw(newYaw);
        player.setPitch(newPitch);

        wantForward = wantBack = wantLeft = wantRight = false;
        if (!aimOnly) {
            switch (a.move) {
                case 0 -> wantForward = true;
                case 1 -> { wantForward = true; wantRight = true; }
                case 2 -> wantRight = true;
                case 3 -> { wantBack = true; wantRight = true; }
                case 4 -> wantBack = true;
                case 5 -> { wantBack = true; wantLeft = true; }
                case 6 -> wantLeft = true;
                case 7 -> { wantForward = true; wantLeft = true; }
                default -> { /* HOLD */ }
            }
            wantJump = a.jump;
            wantSprint = a.sprint;
            wantSneak = a.sneak;
            wantAttack = a.attack;
        } else {
            wantJump = wantSprint = wantSneak = wantAttack = false;
        }

        setKey(client.options.forwardKey, wantForward);
        setKey(client.options.backKey, wantBack);
        setKey(client.options.leftKey, wantLeft);
        setKey(client.options.rightKey, wantRight);
        setKey(client.options.jumpKey, wantJump);
        setKey(client.options.sprintKey, wantSprint);
        setKey(client.options.sneakKey, wantSneak);
        setKey(client.options.attackKey, wantAttack);
    }

    private static void setKey(KeyBinding key, boolean pressed) {
        key.setPressed(pressed);
    }
}