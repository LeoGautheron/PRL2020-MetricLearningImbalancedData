#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize


class iml():
    def __init__(self, num_const, m=1, Lambda=1, randomState=np.random):
        self.num_const = num_const
        self.m = m
        self.Lambda = Lambda
        self.randomState = randomState

    def fit(self, X, Y):
        self.X = X
        self.n = len(X)

        k1 = self.randomState.randint(low=0, high=self.X.shape[0],
                                      size=self.num_const)
        k2 = self.randomState.randint(low=0, high=self.X.shape[0],
                                      size=self.num_const)
        ss = np.where(Y[k1] == Y[k2])[0]
        dd = np.where(Y[k1] != Y[k2])[0]

        self.Sim_i = k1[ss]
        self.Sim_j = k2[ss]
        self.Dis_i = k1[dd]
        self.Dis_j = k2[dd]

        # Call the L-BFGS-B optimizer with the identity matrix as initial point
        L, loss, details = optimize.fmin_l_bfgs_b(
                       maxiter=200, func=self.loss_grad, x0=np.eye(X.shape[1]))

        # Reshape result from optimizer
        self.L_ = L.reshape(X.shape[1], X.shape[1])

    def loss_grad(self, L):
        L = L.reshape((self.X.shape[1], self.X.shape[1]))
        M = L.T.dot(L)

        # Compute pairwise mahalanobis distance between the examples
        # with the current projection matrix L
        Dm_s = np.sum((self.X[self.Sim_i].dot(L.T) -
                       self.X[self.Sim_j].dot(L.T))**2, axis=1)
        Dm_d = np.sum((self.X[self.Dis_i].dot(L.T) -
                       self.X[self.Dis_j].dot(L.T))**2, axis=1)

        # Sim pairs
        idx = np.where(Dm_s > 1)[0]
        diff = self.X[self.Sim_i[idx]] - self.X[self.Sim_j[idx]]
        Sim_g = 2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        Sim_l = np.sum(Dm_s[idx]) - len(idx)  # loss

        # Dis pairs
        idx = np.where(Dm_d < 1 + self.m)[0]
        diff = self.X[self.Dis_i[idx]] - self.X[self.Dis_j[idx]]
        Dis_g = -2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        Dis_l = len(idx)*(1 + self.m) - np.sum(Dm_d[idx])  # loss

        # Squared Frobenius norm term
        identity = np.eye(M.shape[0])
        N_g = 4*L.dot(L.T.dot(L) - identity)  # gradient
        N_l = np.sum((M-identity)**2)  # loss

        loss = (Sim_l +
                Dis_l +
                self.Lambda*N_l)
        gradient = (Sim_g +
                    Dis_g +
                    self.Lambda*N_g)

        return loss, gradient.flatten()

    def transform(self, X):
        return X.dot(self.L_.T)
