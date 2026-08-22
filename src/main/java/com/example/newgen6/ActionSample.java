package com.example.newgen6.rl;

public record ActionSample(BotAction action, int actionIndex, double logProb, double value) {}
