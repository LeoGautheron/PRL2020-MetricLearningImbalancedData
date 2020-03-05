#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import gzip

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


r = []
datasets = []
for filename in glob.glob("./results/res*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    r.append(res["res"])

    if datasets == [] or len(res["datasets"]) < len(datasets):
        datasets = res["datasets"]
    algos = res["algos"]
print(datasets)
# datasets = ['spectfheart']


def getTestResultsValidation(r):
    res = []
    for i in range(len(r)):
        res.append({})
        for da in datasets:
            res[i][da] = {"test": [], "pctExamples": r[i][da]["pctExamples"]}
            for j in range(len(r[0][da]["test"])):
                res[i][da]["test"].append({})
                for a in algos:
                    sor = []
                    for nameP in r[i][da]["valid"][j][a].keys():
                        perfP = []
                        for perf in r[i][da]["valid"][j][a][nameP]:
                            tn, fp, fn, tp = perf["test"]
                            F1 = tp/(tp+fn/2+fp/2)
                            perfP.append(F1)
                        sor.append((np.mean(perfP), nameP))
                    sor = sorted(sor)
                    bestP = sor[-1][1]
                    res[i][da]["test"][j][a] = r[i][da]["test"][j][a][bestP]
    return res


def getMean(r):
    mr = {}
    nb = {}
    for da in datasets:
        mr[da] = {}
        nb[da] = r[0][da]["pctExamples"]
        for a in algos:
            mr[da][a] = {}
            for s in ["train", "test"]:
                mr[da][a][s] = []
                for j in range(len(r[0][da]["test"])):
                    F1s = []
                    for i in range(len(r)):
                        tn, fp, fn, tp = r[i][da]["test"][j][a][s]
                        F1 = tp/(tp+fn/2+fp/2)
                        F1s.append(F1)
                    mr[da][a][s].append(np.mean(F1s))

    all = "Mean results over the 22 datasets"
    mr[all] = {}
    nb[all] = []
    for a in algos:
        mr[all][a] = {}
        for s in ["train", "test"]:
            mr[all][a][s] = []
            for da in datasets:
                for j in range(len(mr[da][a][s])):
                    if j >= len(mr[all][a][s]):
                        mr[all][a][s].append((1, mr[da][a][s][j]))
                    else:
                        cnt, val1 = mr[all][a][s][j]
                        mr[all][a][s][j] = (cnt+1, val1+mr[da][a][s][j])
                    if j >= len(nb[all]):
                        nb[all].append(nb[da][j])
            for j in range(len(mr[all][a][s])):
                cnt, val1 = mr[all][a][s][j]
                mr[all][a][s][j] = (val1 / cnt)

    return mr, nb


def plot(r):
    res = getTestResultsValidation(r)
    mr, nb = getMean(res)

    # algos = ["Euclidean", "GMML", "ITML", "LMNN", "IML"]
    # algos = ["ML1", "ML2", "IML"]
    bar_width = 0.75 / len(algos)

    matplotlib.rcParams['font.size'] = 14
    for da in datasets + ["Mean results over the 22 datasets"]:
        fig = plt.figure(1, figsize=(10, 3))

        ax = fig.add_subplot(1, 1, 1)
        ax.grid(zorder=0)
        ax.set_title(da)
        ax.set_xlabel("Percentage of positive examples")
        ax.set_ylim([0, 100])
        ax.set_yticks(np.arange(0, 100.1, 10))
        ax.set_ylabel("F1-measure")

        index = np.arange(len(nb[da]))
        opacity = 1.0

        ax.set_xticks(index+bar_width*len(algos)/2-bar_width/2)
        labels = [str(int(pct*100)) + "%" for pct in nb[da]]
        ax.set_xticklabels(labels)

        for i, a in enumerate(algos):
            vals = [val*100 for val in mr[da][a]["test"]]
            ax.bar(index+i*bar_width, vals, bar_width,
                   alpha=opacity,
                   label=a, zorder=2)
        ax.legend(ncol=len(algos), loc=9, bbox_to_anchor=(0.5, 1.32))
        plt.savefig("res_" + da + ".pdf", bbox_inches="tight")
        plt.close(fig)


plot(r)
