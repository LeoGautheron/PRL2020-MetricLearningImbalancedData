#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import OrderedDict
import time

import numpy as np

f = "../datasets/"


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def loadAbalone():
    data, n, d = loadCsv(f + 'abalone/abalone.data')

    sex = data[:, 0]
    isMale = (sex == "M").astype(float).reshape(-1, 1)
    isFemale = (sex == "F").astype(float).reshape(-1, 1)
    isInfant = (sex == "I").astype(float).reshape(-1, 1)
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawX = np.hstack((isMale, isFemale, isInfant, rawX))
    rawY = data[:, d-1].astype(int)

    rawY[rawY != 8] = -1
    rawY[rawY == 8] = 1
    return rawX, rawY


def loadAustralian():
    data, n, d = loadCsv(f + 'australian/australian.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadBalance():
    data, n, d = loadCsv(f + 'balance/balance.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]
    rawY = rawY.astype(np.dtype(('U10', 1)))
    rawY[rawY != 'L'] = "-1"
    rawY[rawY == 'L'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadBupa():
    data, n, d = loadCsv(f + 'bupa/bupa.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadGerman():
    data, n, d = loadCsv(f + 'german/german.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadGlass():
    data, n, d = loadCsv(f + 'glass/glass.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadHayes():
    data, n, d = loadCsv(f + 'hayes/hayes-roth.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 3] = -1
    rawY[rawY == 3] = 1
    return rawX, rawY


def loadHeart():
    data, n, d = loadCsv(f + 'heart/heart.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadIonosphere():
    data, n, d = loadCsv(f + 'ionosphere/ionosphere.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'b'] = '-1'
    rawY[rawY == 'b'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadLibras():
    data, n, d = loadCsv(f + 'libras/libras.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)

    rawY[rawY != 1] = -1
    return rawX, rawY


def loadNewthyroid():
    data, n, d = loadCsv(f + 'newthyroid/newthyroid.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY < 2] = -1
    rawY[rawY >= 2] = 1
    return rawX, rawY


def loadPageblocks():
    data, n, d = loadCsv(f + 'pageblocks/pageblocks.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]

    rawY = rawY.astype(int)
    rawY[rawY == 1] = -1
    rawY[rawY == 2] = -1
    rawY[rawY == 3] = 1
    rawY[rawY == 4] = 1
    rawY[rawY == 5] = 1
    return rawX, rawY


def loadPima():
    data, n, d = loadCsv(f + 'pima/pima-indians-diabetes.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSegmentation():
    data, n, d = loadCsv(f + 'segmentation/segmentation.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]

    rawY[rawY == "WINDOW"] = '1'
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSonar():
    data, n, d = loadCsv(f + 'sonar/sonar.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'R'] = '-1'
    rawY[rawY == 'R'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSpambase():
    data, n, d = loadCsv(f + 'spambase/spambase.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadSpectfheart():
    data, n, d = loadCsv(f + 'spectfheart/spectfheart.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 1] = -1
    rawY[rawY == 0] = 1
    return rawX, rawY


def loadSplice():
    data, n, d = loadCsv(f + 'splice/splice.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY == 1] = 2
    rawY[rawY == -1] = 1
    rawY[rawY == 2] = -1
    return rawX, rawY


def loadVehicle():
    data, n, d = loadCsv(f + 'vehicle/vehicle.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != "van"] = '-1'
    rawY[rawY == "van"] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWdbc():
    data, n, d = loadCsv(f + 'wdbc/wdbc.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'M'] = '-1'
    rawY[rawY == 'M'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWine():
    data, n, d = loadCsv(f + 'wine/wine.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadYeast():
    data, n, d = loadCsv(f + 'yeast/yeast.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY == "ME3"] = '1'
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


d = OrderedDict()
s = time.time()
d["hayes"] = loadHayes()                   #    160  19.38%
d["wine"] = loadWine()                     #    178  33.15%
d["sonar"] = loadSonar()                   #    208  46.64%
d["glass"] = loadGlass()                   #    214  32.71%
d["newthyroid"] = loadNewthyroid()         #    215  30.23%
d["spectfheart"] = loadSpectfheart()       #    267  20.60%
d["heart"] = loadHeart()                   #    270  44.44%
d["bupa"] = loadBupa()                     #    345  42.03%
d["iono"] = loadIonosphere()               #    351  35.90%
d["libras"] = loadLibras()                 #    360   6.66%
d["wdbc"] = loadWdbc()                     #    569  37.26%
d["balance"] = loadBalance()               #    625  46.08%
d["australian"] = loadAustralian()         #    690  44.49%
d["pima"] = loadPima()                     #    768  34.90%
d["vehicle"] = loadVehicle()               #    846  23.52%
d["german"] = loadGerman()                 #   1000  30.00%
d["yeast"] = loadYeast()                   #   1484  10.98%
d["segmentation"] = loadSegmentation()     #   2310  14.29%
d["splice"] = loadSplice()                 #   3175  46.64%
d["abalone"] = loadAbalone()               #   4177  13.60%
d["spambase"] = loadSpambase()             #   4597  39.42%
d["pageblocks"] = loadPageblocks()         #   5473   4.22%
print("Data loaded in {:5.2f}".format(time.time()-s))
