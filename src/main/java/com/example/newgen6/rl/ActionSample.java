package com.example.newgen6.rl;

public final class ActionSample {
    public final boolean forward;
    public final boolean backward;
    public final boolean left;
    public final boolean right;
    public final boolean jump;
    public final boolean sprint;
    public final boolean sneak;
    public final boolean attack;

    public final float mouseX;
    public final float mouseY;

    public final float logProb;
    public final float value;

    public final float[] vector = new float[AgentConfig.ACTION_DIM];

    public ActionSample(boolean[] keys, float mouseX, float mouseY, float logProb, float value) {
        this.forward = keys[0];
        this.backward = keys[1];
        this.left = keys[2];
        this.right = keys[3];
        this.jump = keys[4];
        this.sprint = keys[5];
        this.sneak = keys[6];
        this.attack = keys[7];

        this.mouseX = mouseX;
        this.mouseY = mouseY;

        this.logProb = logProb;
        this.value = value;

        vector[0] = forward ? 1.0f : 0.0f;
        vector[1] = backward ? 1.0f : 0.0f;
        vector[2] = left ? 1.0f : 0.0f;
        vector[3] = right ? 1.0f : 0.0f;
        vector[4] = jump ? 1.0f : 0.0f;
        vector[5] = sprint ? 1.0f : 0.0f;
        vector[6] = sneak ? 1.0f : 0.0f;
        vector[7] = attack ? 1.0f : 0.0f;
        vector[8] = mouseX;
        vector[9] = mouseY;
    }
}