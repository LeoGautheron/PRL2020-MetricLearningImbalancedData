#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Information-Theoretic Metric Learning
# Jason V. Davis, Brian Kulis, Prateek Jain, Suvrit Sra, Inderjit S. Dhillon
# International Conference on Machine Learning
# 2007

from numpy.linalg import cholesky
import numpy as np


def pairs(Y, nbConstraints, sameLabel=True, maxIter=10, randomState=np.random):
    nbLabels = len(Y)
    ab = set()
    it = 0
    while it < maxIter and len(ab) < nbConstraints:
        nc = nbConstraints - len(ab)
        for aidx in randomState.randint(nbLabels, size=nc):
            if sameLabel:
                mask = Y[aidx] == Y
                mask[aidx] = False  # avoid identity pairs
            else:
                mask = Y[aidx] != Y
            b_choices, = np.where(mask)
            if len(b_choices) > 0:
                ab.add((aidx, randomState.choice(b_choices)))
        it += 1
    ab = np.array(list(ab)[:nbConstraints], dtype=int)
    labels_idx = np.arange(len(Y))
    return labels_idx[ab.T]


def positiveNegativePairs(Y, nbConstraints, randomState=np.random):
    Si, Sj = pairs(Y, nbConstraints, sameLabel=True, randomState=randomState)
    Di, Dj = pairs(Y, nbConstraints, sameLabel=False, randomState=randomState)
    return Si, Sj, Di, Dj


class ITML():
    def __init__(self, boundU=None, boundL=None, A0=None, gamma=1.,
                 nbConstraints=None, maxIter=1000, convergenceThreshold=1e-3,
                 randomState=np.random):
        self.boundU = boundU
        self.boundL = boundL
        self.A0 = A0
        self.gamma = gamma
        self.nbConstraints = nbConstraints
        self.maxIter = maxIter
        self.convergenceThreshold = convergenceThreshold
        self.randomState = randomState

    def fit(self, X, Y):
        if self.nbConstraints is None:
            nbClasses = len(np.unique(Y))
            self.nbConstraints = 20 * nbClasses**2
        Si, Sj, Di, Dj = positiveNegativePairs(
                           Y, self.nbConstraints, randomState=self.randomState)
        # check to make sure that no two constrained vectors are identical
        noIdent = np.apply_along_axis(np.linalg.norm, 1, X[Si] - X[Sj]) > 1e-9
        Si, Sj = Si[noIdent], Sj[noIdent]
        noIdent = np.apply_along_axis(np.linalg.norm, 1, X[Di] - X[Dj]) > 1e-9
        Di, Dj = Di[noIdent], Dj[noIdent]
        # init bounds
        if self.boundU is None or self.boundL is None:
            nb = X.shape[0] // 10
            if nb % 2 == 1:
                nb += 1
            idx = self.randomState.randint(0, len(X), max(1000, nb))
            dists = [np.linalg.norm(X[idx[i]] - X[idx[i+1]])
                     for i in range(0, len(idx), 2)]
            bins = np.histogram(dists, 100)
            self.boundU = bins[1][4]
            self.boundL = bins[1][94]
        self.boundU = 1e-9 if self.boundU == 0 else self.boundU
        self.boundL = 1e-9 if self.boundL == 0 else self.boundL
        constraints = np.vstack((X[Si] - X[Sj], X[Di] - X[Dj]))
        gamma = self.gamma
        nbSimilar = len(Si)
        nbDissimilar = len(Di)

        self.A = np.identity(X.shape[1]) if self.A0 is None else self.A0  # 1
        lambd = np.zeros(nbSimilar + nbDissimilar)  # 1
        lambdOld = np.zeros_like(lambd)
        xi = np.hstack((np.zeros(nbSimilar) + self.boundU,
                        np.zeros(nbDissimilar) + self.boundL))  # 2
        for it in range(self.maxIter):  # 3
            for i, xi_minus_xj in enumerate(constraints):  # 3.1
                p = xi_minus_xj.dot(self.A).dot(xi_minus_xj)  # 3.2
                delta = 1 if i < nbSimilar else -1  # 3.3
                alpha = min(lambd[i], delta/2*(1./p - gamma/xi[i]))  # 3.4
                beta = delta*alpha/(1 - delta*alpha*p)  # 3.5
                xi[i] = 1./((1 / xi[i]) + (alpha*delta / gamma))  # 3.6
                lambd[i] -= alpha  # 3.7
                Av = self.A.dot(xi_minus_xj)
                self.A += np.outer(Av, Av * beta)  # 3.8

            normsum = np.linalg.norm(lambd) + np.linalg.norm(lambdOld)
            if normsum == 0:
                conv = np.inf
                break
            conv = np.abs(lambdOld - lambd).sum() / normsum
            if conv < self.convergenceThreshold:  # 4
                break
            lambdOld = lambd.copy()
        self.n_iter_ = it
        self.L_ = cholesky(self.A).T
        return self

    def transform(self, X=None):
        return X.dot(self.L_.T)
