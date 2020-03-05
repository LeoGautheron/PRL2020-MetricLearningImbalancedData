#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import optimize


class iml():
    def __init__(self, pClass, k=1, m=1, Lambda=1, randomState=np.random):
        self.pClass = pClass  # minority<=>positive class
        self.k = k
        self.m = m
        self.Lambda = Lambda
        self.randomState = randomState

    def fit(self, X, Y):
        self.X = X

        self.idxP = np.where(Y == self.pClass)[0]  # indexes of pos examples
        self.idxN = np.where(Y != self.pClass)[0]  # indexes of other examples
        self.Np = len(self.idxP)
        self.Nn = len(self.idxN)
        self.n = len(X)

        if self.Np <= 1:
            print("Error, there should be at least 2 positive examples")
            return

        # Initialize the number of neighbors
        if self.k >= self.Np:
            self.k = self.Np - 1  # maximum possible number of neighbors
        if self.k <= 0:
            self.k = 1  # we need at least one neighbor

        # Positive Positive Pairs
        D = euclidean_distances(self.X[self.idxP], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimP_i = []
        self.SimP_j = []
        for idxI in range(len(self.idxP)):  # for each positive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimP_i.append(self.idxP[idxI])
                self.SimP_j.append(self.idxP[idxJ])
                idxIdxJ += 1
        self.SimP_i = np.array(self.SimP_i)
        self.SimP_j = np.array(self.SimP_j)

        # Negative Negative Pairs
        D = euclidean_distances(self.X[self.idxN], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimN_i = []
        self.SimN_j = []
        for idxI in range(len(self.idxN)):  # for each negative example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimN_i.append(self.idxN[idxI])
                self.SimN_j.append(self.idxN[idxJ])
                idxIdxJ += 1
        self.SimN_i = np.array(self.SimN_i)
        self.SimN_j = np.array(self.SimN_j)

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
        Dm_pp = np.sum((self.X[self.SimP_i].dot(L.T) -
                        self.X[self.SimP_j].dot(L.T))**2, axis=1)
        Dm_nn = np.sum((self.X[self.SimN_i].dot(L.T) -
                        self.X[self.SimN_j].dot(L.T))**2, axis=1)

        # Sim+ (Positive, Positive) pairs
        idx = np.where(Dm_pp > 1)[0]
        diff = self.X[self.SimP_i[idx]] - self.X[self.SimP_j[idx]]
        SimP_g = 2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        SimP_l = np.sum(Dm_pp[idx]) - len(idx)  # loss

        # Compute pairwise mahalanobis Dm distance between examples self.X
        # with the current projection matrix L
        LXp = self.X[self.idxP].dot(L.T)
        LXn = self.X[self.idxN].dot(L.T)
        Dmpn = euclidean_distances(LXp, LXn, squared=True)  # between pos & neg

        # D^+ term
        idxs = np.argpartition(Dmpn, self.k)[:, :self.k]  # each + k smallest
        rows = np.repeat(np.arange(len(idxs)), self.k)
        cols = idxs.flatten()
        i1, i2 = np.where(Dmpn[rows, cols].reshape(-1, self.k) < 1 + self.m)
        diff = self.X[self.idxP[i1]]-self.X[self.idxN[idxs[i1, i2]]]
        DisP_g = -2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        DisP_l = len(i1)*(1 + self.m) - np.sum(Dmpn[i1, idxs[i1, i2]])

        # D^- term
        idxs = np.argpartition(Dmpn.T, self.k)[:, :self.k]  # each - k smallest
        rows = np.repeat(np.arange(len(idxs)), self.k)
        cols = idxs.flatten()
        i1, i2 = np.where(Dmpn.T[rows, cols].reshape(-1, self.k) < 1 + self.m)
        diff = self.X[self.idxN[i1]]-self.X[self.idxP[idxs[i1, i2]]]
        DisN_g = -2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        DisN_l = len(i1)*(1 + self.m) - np.sum(Dmpn.T[i1, idxs[i1, i2]])

        # Sim- (Negative, Negative) pairs
        idx = np.where(Dm_nn > 1)[0]
        diff = self.X[self.SimN_i[idx]] - self.X[self.SimN_j[idx]]
        SimN_g = 2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        SimN_l = np.sum(Dm_nn[idx]) - len(idx)  # loss

        # Squared Frobenius norm term
        identity = np.eye(M.shape[0])
        N_g = 4*L.dot(L.T.dot(L) - identity)  # gradient
        N_l = np.sum((M-identity)**2)  # loss

        loss = ((self.Nn/self.n)*SimP_l +
                (self.Nn/self.n)*DisP_l +
                (self.Np/self.n)*DisN_l +
                (self.Np/self.n)*SimN_l +
                self.Lambda*N_l)
        gradient = ((self.Nn/self.n)*SimP_g +
                    (self.Nn/self.n)*DisP_g +
                    (self.Np/self.n)*DisN_g +
                    (self.Np/self.n)*SimN_g +
                    self.Lambda*N_g)
        return loss, gradient.flatten()

    def transform(self, X):
        return X.dot(self.L_.T)
