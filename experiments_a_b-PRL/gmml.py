#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.linalg import cholesky
from scipy.linalg import fractional_matrix_power
import numpy as np


class GeometricMeanMetricLearning():
    def __init__(self, t=0.5, randomState=None):
        self.t = t
        self.randomState = randomState

    def fit(self, X, y):
        n = len(X)
        hk = len(np.unique(y))
        num_const = 40 * (hk * (hk-1))

        # Generate constraints
        k1 = self.randomState.randint(low=0, high=n, size=num_const)
        k2 = self.randomState.randint(low=0, high=n, size=num_const)
        ss = np.where(y[k1] == y[k2])[0]
        dd = np.where(y[k1] != y[k2])[0]
        SD = X[k1[ss], :] - X[k2[ss], :]
        DD = X[k1[dd], :] - X[k2[dd], :]
        S = SD.T.dot(SD)
        D = DD.T.dot(DD)

        # Regularization to obtain A PSD
        S += 1 * np.eye(len(S))
        D += 1 * np.eye(len(D))

        # Compute metric
        self.A = np.real(np.linalg.solve(S, fractional_matrix_power(
                                                            S.dot(D), self.t)))
        self.L_ = cholesky(self.A).T

    def transform(self, X=None):
        return X.dot(self.L_.T)
