#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import pickle
import random
import time
import os
import sys
import gzip

from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from IML import iml as IML
from gmml import GeometricMeanMetricLearning as GMML
from lmnn import LargeMarginNearestNeighbor as LMNN
from itml import ITML
from imls import imls
import datasets

if len(sys.argv) == 2:
    np.random.seed(int(sys.argv[1]))
    random.seed(int(sys.argv[1]))

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

minClass = +1  # label of minority class
majClass = -1  # label of majority class

K = 3  # for KNN classifier

nbFoldValid = 5
maxNbParamTested = 100

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################
if not os.path.exists("results"):
    try:
        os.makedirs("results")
    except:
        pass


def knn(k, Xtrain, Ytrain, Xtest):
    """
    Classic kNN function. Take as input train features and labels. And
    test features. Then compute pairwise distances between test and train.
    And for each test example, return the majority class among its kNN.
    """
    d = euclidean_distances(Xtest, Xtrain, squared=True)
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtest.shape[0], k)
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)


def knnSame(k, Xtrain, Ytrain):
    """
    A varriant of kNN. Here, we want to use the same set to learn and predict.
    We compute pairwise distances between train and itself. Then we fill the
    diagonal of this square matrix to be infinite to avoid having one
    example being among his neighbors. The class prediction is then the same
    as the classic kNN. Without adding infinite on the diagonal, we would
    obtain 0% error by looking only at the 1nearest neighbor.
    """
    d = euclidean_distances(Xtrain, squared=True)
    np.fill_diagonal(d, np.inf)
    nnc = Ytrain[np.argsort(d)[..., :k].flatten()].reshape(Xtrain.shape[0], k)
    pred = [max(nnc[i], key=Counter(nnc[i]).get) for i in range(nnc.shape[0])]
    return np.array(pred)


def listP(dic, shuffle=False):
    """
    Input: dictionnary with parameterName: array parameterRange
    Output: list of dictionnary with parameterName: parameterValue
    """
    # Recover the list of parameter names.
    params = list(dic.keys())
    # Initialy, the list of parameter to use is the list of values of
    # the first parameter.
    listParam = [{params[0]: value} for value in dic[params[0]]]
    # For each parameter p after the first, the "listParam" contains a
    # number x of dictionnary. p can take y possible values.
    # For each value of p, create x parameter by adding the value of p in the
    # dictionnary. After processing parameter p, our "listParam" is of size x*y
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    if shuffle:
        random.shuffle(listParam)
    return listParam


def applyAlgo(algo, p, Xtrain, Ytrain, Xtest, Ytest):
    if a.startswith("o"):
        nbMinority = len(Xtrain[Ytrain == minClass])
        if nbMinority <= 5:
            sm = SMOTE(random_state=42, k_neighbors=nbMinority-1)
        else:
            sm = SMOTE(random_state=42)
        Xtrain2, Ytrain2 = sm.fit_sample(Xtrain, Ytrain)
    elif a.startswith("u"):
        rus = RandomUnderSampler(random_state=42)
        Xtrain2, Ytrain2 = rus.fit_sample(Xtrain, Ytrain)
    else:
        Xtrain2, Ytrain2 = Xtrain, Ytrain

    scaler = StandardScaler()
    scaler.fit(Xtrain2)
    Xtrain2 = scaler.transform(Xtrain2)
    Xtest = scaler.transform(Xtest)

    if algo.endswith("IMLS"):
        ml = imls(k=K, mu=p["mu"], randomState=np.random.RandomState(1))
        Ytest_pred = ml.fitPredict(Xtrain2, Ytrain2, Xtest)
        perf = {}
        for true, pred, name in [(Ytest, Ytest_pred, "train"),
                                 (Ytest, Ytest_pred, "test")]:
            tn, fp, fn, tp = confusion_matrix(
                               true, pred, labels=[majClass, minClass]).ravel()
            perf[name] = ((int(tn), int(fp), int(fn), int(tp)))
        return perf

    if algo.endswith("IML"):
        ml = IML(pClass=minClass, k=K, m=p["m"], Lambda=p["Lambda"])
    elif algo.endswith("LMNN"):
        ml = LMNN(k=K, mu=p["mu"], randomState=np.random.RandomState(1))
    elif algo.endswith("GMML"):
        ml = GMML(t=p["t"], randomState=np.random.RandomState(1))
    elif algo.endswith("ITML"):
        ml = ITML(gamma=p["gamma"], randomState=np.random.RandomState(1))

    if not algo.endswith("Euclidean"):
        ml.fit(Xtrain2, Ytrain2)
        Xtrain2 = ml.transform(Xtrain2)
        Xtest = ml.transform(Xtest)

    # Apply kNN to predict classes of test examples
    Ytrain_pred = knnSame(K, Xtrain2, Ytrain2)
    Ytest_pred = knn(K, Xtrain2, Ytrain2, Xtest)

    perf = {}
    for true, pred, name in [(Ytrain2, Ytrain_pred, "train"),
                             (Ytest, Ytest_pred, "test")]:
        # Compute performance measures by comparing prediction with true labels
        tn, fp, fn, tp = confusion_matrix(true, pred,
                                          labels=[majClass, minClass]).ravel()
        perf[name] = ((int(tn), int(fp), int(fn), int(tp)))

    return perf


###############################################################################
# Definition of parameters to test during the cross-validation for each algo
listParams = {}
listParams["Euclidean"] = listP({"None": [None]})
listParams["LMNN"] = listP({"mu": np.arange(0, 1.01, 0.05)})
listParams["ITML"] = listP({"gamma": [2**i for i in range(-10, 11)]})
listParams["GMML"] = listP({"t": np.arange(0, 1.01, 0.05)})
listParams["IMLS"] = listP({"mu": np.arange(0, 1.01, 0.05)})
listParams["IML"] = listP({"m": [1, 10, 100, 1000, 10000],
                           "Lambda": [0, 0.01, 0.1, 1, 10]})

listParams["oEuclidean"] = listParams["Euclidean"]
listParams["oLMNN"] = listParams["LMNN"]
listParams["oITML"] = listParams["ITML"]
listParams["oGMML"] = listParams["GMML"]
listParams["oIMLS"] = listParams["IMLS"]
listParams["oIML"] = listParams["IML"]

listParams["uEuclidean"] = listParams["Euclidean"]
listParams["uLMNN"] = listParams["LMNN"]
listParams["uITML"] = listParams["ITML"]
listParams["uGMML"] = listParams["GMML"]
listParams["uIMLS"] = listParams["IMLS"]
listParams["uIML"] = listParams["IML"]

listNames = {a: [] for a in listParams.keys()}
listParametersNames = {a: {} for a in listParams.keys()}
for a in listParams.keys():
    for i, p in enumerate(listParams[a]):
        listParametersNames[a][str(p)] = p
        listNames[a].append(str(p))

r = {}  # All the results are stored in this dictionnary
datasetsDone = []
startTime = time.time()
for da in datasets.d.keys():  # For each dataset
    print(da)
    X, y = datasets.d[da][0], datasets.d[da][1]

    if len(sys.argv) == 2:
        np.random.seed(int(sys.argv[1]))
        random.seed(int(sys.argv[1]))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle=True,
                                                    stratify=y, test_size=0.7)

    skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
    foldsTrainValid = list(skf.split(Xtrain, ytrain))

    r[da] = {"valid": {a: {} for a in listParametersNames.keys()},
             "test": {a: {} for a in listParametersNames.keys()}}
    for a in listParametersNames.keys():  # For each algo
        nbParamToTest = len(listParametersNames[a])
        nbParamTested = 0
        for nameP in listNames[a]:  # For each set of parameters
            p = listParametersNames[a][nameP]
            r[da]["valid"][a][nameP] = []
            # Compute performance on each validation fold
            for iFoldVal in range(nbFoldValid):
                fTrain, fValid = foldsTrainValid[iFoldVal]
                perf = applyAlgo(a, p,
                                 Xtrain[fTrain], ytrain[fTrain],
                                 Xtrain[fValid], ytrain[fValid])
                r[da]["valid"][a][nameP].append(perf)
            nbParamTested += 1
            # Compute performance on test set by training on the union of
            # all the validation folds.
            perf = applyAlgo(a, p, Xtrain, ytrain, Xtest, ytest)
            r[da]["test"][a][nameP] = perf
            tn, fp, fn, tp = perf["test"]
            F1 = tp/(tp+fn/2+fp/2)
            print(da, a,
                  str(nbParamTested)+"/"+str(nbParamToTest),
                  "time: {:8.2f} sec".format(time.time()-startTime),
                  "test F1 {:5.2f}".format(F1*100), p)
            if nbParamTested >= maxNbParamTested:
                break

    datasetsDone.append(da)
    # Save the results at the end of each dataset
    if len(sys.argv) == 2:
        f = gzip.open("./results/res" + sys.argv[1] + ".pklz", "wb")
    else:
        f = gzip.open("./results/res" + str(startTime) + ".pklz", "wb")
    pickle.dump({"res": r, "algos": list(listParametersNames.keys()),
                 "datasets": datasetsDone}, f)
    f.close()
