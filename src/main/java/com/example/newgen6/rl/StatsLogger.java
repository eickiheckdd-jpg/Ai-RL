package com.example.newgen6.rl;

import java.io.*;
import java.nio.file.*;
import java.util.Locale;

/**
 * Tracks training statistics so you can tell if the agent is actually learning.
 */
public final class StatsLogger {

    private static final Path LOG_DIR = Paths.get("newgen6_logs");
    private final Path csvPath;

    // running episode stats
    private float epReward = 0f;
    private int epSteps = 0;
    private float epDmgDealt = 0f;
    private float epDmgTaken = 0f;
    private int epHits = 0;
    private int epCrits = 0;
    private int wins = 0, losses = 0, draws = 0;

    // PPO losses (set from trainer)
    private float lastPolicyLoss = 0f;
    private float lastValueLoss = 0f;
    private float lastEntropy = 0f;
    private float lastKL = 0f;

    private long totalEpisodes = 0;

    public StatsLogger() {
        try {
            Files.createDirectories(LOG_DIR);
            csvPath = LOG_DIR.resolve("training.csv");
            if (!Files.exists(csvPath)) {
                Files.writeString(csvPath,
                        "episode,steps,reward,dmg_dealt,dmg_taken,hits,crits,win,policy_loss,value_loss,entropy,kl,stage\n");
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void onStep(float reward, float dmgDealt, float dmgTaken, boolean hit, boolean crit) {
        epReward += reward;
        epSteps++;
        epDmgDealt += dmgDealt;
        epDmgTaken += dmgTaken;
        if (hit) epHits++;
        if (crit) epCrits++;
    }

    public void endEpisode(boolean won, boolean lost, int stage) {
        totalEpisodes++;
        if (won) wins++;
        else if (lost) losses++;
        else draws++;

        try (BufferedWriter w = Files.newBufferedWriter(csvPath, StandardOpenOption.APPEND)) {
            w.write(String.format(Locale.US,
                    "%d,%d,%.4f,%.2f,%.2f,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%d%n",
                    totalEpisodes, epSteps, epReward, epDmgDealt, epDmgTaken,
                    epHits, epCrits, won ? 1 : 0,
                    lastPolicyLoss, lastValueLoss, lastEntropy, lastKL, stage));
        } catch (IOException ignored) {}

        // reset episode
        epReward = 0f;
        epSteps = 0;
        epDmgDealt = epDmgTaken = 0f;
        epHits = epCrits = 0;
    }

    public void setLosses(float policy, float value, float entropy, float kl) {
        lastPolicyLoss = policy;
        lastValueLoss = value;
        lastEntropy = entropy;
        lastKL = kl;
    }

    // getters for HUD
    public float getEpReward() { return epReward; }
    public int getEpSteps() { return epSteps; }
    public float getEpDmgDealt() { return epDmgDealt; }
    public float getEpDmgTaken() { return epDmgTaken; }
    public int getWins() { return wins; }
    public int getLosses() { return losses; }
    public long getTotalEpisodes() { return totalEpisodes; }
    public float getLastPolicyLoss() { return lastPolicyLoss; }
    public float getLastValueLoss() { return lastValueLoss; }
    public float getLastEntropy() { return lastEntropy; }
    public float getWinRate() {
        long t = wins + losses;
        return t == 0 ? 0f : (float) wins / t;
    }
}
