package com.example.newgen6;

import com.example.newgen6.ActionExecutor;
import com.example.newgen6.ActionType;
import com.example.newgen6.RewardCalculator;
import com.example.newgen6.StateExtractor;
import com.example.newgen6.PPOAgent;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.player.PlayerEntity;

import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

public class PvpBotController {

    private final StateExtractor stateExtractor = new StateExtractor();
    private final ActionExecutor actionExecutor = new ActionExecutor();
    private final RewardCalculator rewardCalc = new RewardCalculator();
    private final PPOAgent agent = new PPOAgent(
            StateExtractor.STATE_SIZE,
            ActionType.VALUES.length,
            new int[]{64, 64});

    // Ticks of artificial delay before the agent reacts to what it "saw" -
    // a crude stand-in for human reaction time (~150ms at 20 tps).
    private static final int REACTION_DELAY_TICKS = 3;
    private final Deque<double[]> stateHistory = new ArrayDeque<>();

    private double[] pendingState = null;
    private int pendingAction = -1;
    private double pendingLogProb = 0;
    private double pendingValue = 0;

    public boolean enabled = false;
    private int tickCounter = 0;
    private static final File SAVE_FILE = new File("newgen6_ppo_weights.bin");

    public void onTick(MinecraftClient client) {
        if (!enabled || client.player == null || client.world == null) return;

        PlayerEntity opponent = stateExtractor.findOpponent(client);

        // Close the loop on the action chosen last tick before picking a new one.
        if (pendingState != null) {
            boolean selfDied = client.player.isDead();
            boolean opponentDied = (opponent == null);
            double reward = rewardCalc.computeReward(client.player, opponent, selfDied, opponentDied);
            boolean done = selfDied || opponentDied;

            agent.remember(pendingState, pendingAction, pendingLogProb, pendingValue, reward, done);

            if (done) {
                pendingState = null;
                stateHistory.clear();
                rewardCalc.reset(client.player, opponent);
                return;
            }
        } else {
            rewardCalc.reset(client.player, opponent);
        }

        if (opponent == null) {
            actionExecutor.resetMovementKeys(client);
            return;
        }

        double[] currentState = stateExtractor.extract(client, opponent);
        stateHistory.addLast(currentState);
        if (stateHistory.size() <= REACTION_DELAY_TICKS) {
            actionExecutor.resetMovementKeys(client);
            return; // still filling the reaction-delay buffer
        }
        double[] decisionState = stateHistory.pollFirst();

        PPOAgent.ActResult result = agent.act(decisionState);
        ActionType action = ActionType.VALUES[result.action];
        actionExecutor.apply(client, action, opponent);

        pendingState = decisionState;
        pendingAction = result.action;
        pendingLogProb = result.logProb;
        pendingValue = result.value;

        tickCounter++;
        if (tickCounter % 6000 == 0) { // roughly every 5 minutes at 20 tps
            saveWeights();
        }
    }

    public void saveWeights() {
        try {
            agent.save(SAVE_FILE);
            System.out.println("[newgen6] Saved PPO weights to " + SAVE_FILE.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadWeights() {
        try {
            if (SAVE_FILE.exists()) {
                agent.load(SAVE_FILE);
                System.out.println("[newgen6] Loaded PPO weights.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
