from cgi import test
from curses import KEY_REPLACE
import numpy as np
import pandas as pd
import scipy.sparse as scp
import scipy.special

class LogisticRegressionClassifier:

    def __init__(self, m, k, n, eta, lamb, delta, X, x_col_sums, Y, W):
        # m = number of examples
        self.m = m
        # k = number of classes
        self.k = k
        # n = number of attributes per example
        self.n = n
        # eta = learning rate
        self.eta = eta
        # lamb = penalty term
        self.lamb = lamb
        # delta = k * m matrix
        self.delta = delta
        # X = m * (n+1) matrix of examples where first column is all 1s and the rest are the attributes
        self.X = X
        self.X_tran = X.T
        self.x_col_sums = x_col_sums
        # Y = m * 1 vector of true classifications
        self.Y = Y
        # W = k * n+1 matrix
        self.W = W

    def update_step(self):
        prob_Y_WX = np.exp(np.matmul(self.W, self.X_tran))
        # prob_Y_WX[-1, :] = 1
        prob_Y_WX_norm = prob_Y_WX/prob_Y_WX.sum(axis=0, keepdims=True)

        term1 = self.delta - prob_Y_WX_norm
        term2 = self.lamb * self.W
        term1_m = np.matmul(term1, self.X)

        W_t_next = self.W + (self.eta * (term1_m - term2))
        # W_t_next[:, -1] = 0
        self.W = W_t_next

    def create_weights(self, iterations):
        for i in range(0, iterations):
            self.update_step()

        return self.W

    def classify(self, test_data):
        print('Multiplying matrices...')
        test_matrix = test_data.to_numpy()
        ones_column = np.array([np.ones(test_matrix.shape[0], dtype=float)])

        trimmed_matrix = test_matrix[:, 1:]
        normal_matrix = (trimmed_matrix.T / self.x_col_sums[:, np.newaxis]).T
        # test_matrix[:, 0] = 1
        # normal_data = test_matrix/test_matrix.sum(axis=1, keepdims=True)
        # normal_data[:, 0] = 1

        normal_matrix = np.concatenate((ones_column.T, normal_matrix), axis=1)
        classified_matrix = np.matmul(normal_matrix, self.W.T)
        print(classified_matrix)
        
        print('Finding maximums...')
        classes = np.argmax(classified_matrix, axis=1) + 1
        return classes