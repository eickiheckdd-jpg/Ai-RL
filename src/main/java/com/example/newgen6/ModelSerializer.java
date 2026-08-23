package com.example.newgen6;

import java.io.*;

public class ModelSerializer {
    
    public static void saveModel(DDPGAgent agent, String filepath) {
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filepath)))) {
            writeNetwork(dos, agent.actor.w1, agent.actor.b1, agent.actor.w2, agent.actor.b2);
            writeNetwork(dos, agent.critic.w1, agent.critic.b1, agent.critic.w2, agent.critic.b2);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void loadModel(DDPGAgent agent, String filepath) {
        File file = new File(filepath);
        if (!file.exists()) return;

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filepath)))) {
            readNetwork(dis, agent.actor.w1, agent.actor.b1, agent.actor.w2, agent.actor.b2);
            readNetwork(dis, agent.critic.w1, agent.critic.b1, agent.critic.w2, agent.critic.b2);
            
            // Hard copy to target networks
            System.arraycopy(agent.actor.w1, 0, agent.targetActor.w1, 0, agent.actor.w1.length);
            System.arraycopy(agent.critic.w1, 0, agent.targetCritic.w1, 0, agent.critic.w1.length);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeNetwork(DataOutputStream dos, float[] w1, float[] b1, float[] w2, float[] b2) throws IOException {
        for (float v : w1) dos.writeFloat(v);
        for (float v : b1) dos.writeFloat(v);
        for (float v : w2) dos.writeFloat(v);
        for (float v : b2) dos.writeFloat(v);
    }

    private static void readNetwork(DataInputStream dis, float[] w1, float[] b1, float[] w2, float[] b2) throws IOException {
        for (int i = 0; i < w1.length; i++) w1[i] = dis.readFloat();
        for (int i = 0; i < b1.length; i++) b1[i] = dis.readFloat();
        for (int i = 0; i < w2.length; i++) w2[i] = dis.readFloat();
        for (int i = 0; i < b2.length; i++) b2[i] = dis.readFloat();
    }
}
