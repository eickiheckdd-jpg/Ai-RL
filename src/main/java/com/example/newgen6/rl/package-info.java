/**
 * Pure-Java PPO implementation for NEWGEN6.
 *
 * Key design goals:
 * - Stay under ~2 GB total with Minecraft
 * - 229-feature observation tailored for Sword HT3
 * - Curriculum that starts Neutral (high noise → visible shaking)
 *   and progressively tightens look scale + reduces entropy
 *   so the agent learns faster, smoother aiming over time
 * - Dense rewards for spacing, crits, combos, survival
 */
package com.example.newgen6.rl;
