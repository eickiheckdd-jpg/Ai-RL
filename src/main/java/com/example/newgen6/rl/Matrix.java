package com.example.newgen6.rl;

import java.util.Random;

public class Matrix {
    public int rows, cols;
    public float[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
    }

    public static Matrix randomXavier(int rows, int cols, Random rng) {
        Matrix m = new Matrix(rows, cols);
        float limit = (float) Math.sqrt(6.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = (rng.nextFloat() * 2 * limit) - limit;
            }
        }
        return m;
    }

    // Performs per-parameter adaptive updates using Adam momentum tracking
    public void adamUpdate(Matrix grad, Matrix m, Matrix v, int t, float lr, float beta1, float beta2, float eps) {
        float b1Correction = 1.0f - (float) Math.pow(beta1, t);
        float b2Correction = 1.0f - (float) Math.pow(beta2, t);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float g = grad.data[i][j];
                
                // First moment vector update
                m.data[i][j] = beta1 * m.data[i][j] + (1.0f - beta1) * g;
                
                // Second moment vector update
                v.data[i][j] = beta2 * v.data[i][j] + (1.0f - beta2) * (g * g);

                // Bias corrections
                float mHat = m.data[i][j] / b1Correction;
                float vHat = v.data[i][j] / b2Correction;

                // Gradient update
                this.data[i][j] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }
}