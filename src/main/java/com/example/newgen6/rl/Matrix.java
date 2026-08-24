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

    public static Matrix randomOrthogonal(int rows, int cols, float gain, Random rng) {
        Matrix m = new Matrix(rows, cols);
        float limit = (float) (gain * Math.sqrt(6.0 / (rows + cols)));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = (rng.nextFloat() * 2.0f * limit) - limit;
            }
        }
        return m;
    }

    public void adamUpdate(Matrix grad, Matrix m, Matrix v, int step, float lr, float beta1, float beta2, float eps) {
        float b1Correction = 1.0f - (float) Math.pow(beta1, step);
        float b2Correction = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float g = grad.data[i][j];

                m.data[i][j] = beta1 * m.data[i][j] + (1.0f - beta1) * g;
                v.data[i][j] = beta2 * v.data[i][j] + (1.0f - beta2) * (g * g);

                float mHat = m.data[i][j] / b1Correction;
                float vHat = v.data[i][j] / b2Correction;

                this.data[i][j] -= lr * mHat / ((float) Math.sqrt(vHat) + 1e-5f);
            }
        }
    }
}
