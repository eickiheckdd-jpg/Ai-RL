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
}
