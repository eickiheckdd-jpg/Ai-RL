package com.example.newgen6;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

public class PPOTrainerThread implements Runnable {
    private final PPOAgent agent;
    private final ConcurrentLinkedQueue<List<RolloutBuffer.Step>> trainingQueue = new ConcurrentLinkedQueue<>();
    private volatile boolean running = true;

    public PPOTrainerThread(PPOAgent agent) {
        this.agent = agent;
    }

    public void enqueueRollout(List<RolloutBuffer.Step> steps) {
        trainingQueue.add(steps);
    }

    @Override
    public void run() {
        while (running) {
            try {
                List<RolloutBuffer.Step> steps = trainingQueue.poll();
                if (steps != null && !steps.isEmpty()) {
                    trainOnBatch(steps);
                } else {
                    Thread.sleep(10);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void trainOnBatch(List<RolloutBuffer.Step> steps) {
        int batchSize = steps.size();
        INDArray obsBatch = Nd4j.create(batchSize, RLConfig.OBS_DIM);
        INDArray targetPolicy = Nd4j.create(batchSize, RLConfig.ACTION_CONTINUOUS_DIM);
        INDArray targetValue = Nd4j.create(batchSize, 1);

        for (int i = 0; i < batchSize; i++) {
            RolloutBuffer.Step s = steps.get(i);
            obsBatch.putRow(i, Nd4j.create(s.obs));
            
            targetValue.putScalar(new int[]{i, 0}, s.reward);
            targetPolicy.putScalar(new int[]{i, 0}, s.continuousAction[0] / RLConfig.MAX_YAW_DELTA);
            targetPolicy.putScalar(new int[]{i, 1}, s.continuousAction[1] / RLConfig.MAX_PITCH_DELTA);
        }

        ComputationGraph net = agent.getNetwork();
        synchronized (net) {
            for (int epoch = 0; epoch < RLConfig.N_EPOCHS; epoch++) {
                net.fit(new INDArray[]{obsBatch}, new INDArray[]{targetPolicy, targetValue});
            }
        }
    }

    public void stop() { this.running = false; }
}
