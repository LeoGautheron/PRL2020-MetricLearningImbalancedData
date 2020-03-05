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
                    Accuracies = []
                    for i in range(len(r)):
                        tn, fp, fn, tp = r[i][da]["test"][j][a][s]
                        F1 = tp/(tp+fn/2+fp/2)
                        Accuracy = (tp+tn)/(tp+tn+fp+fn)
                        F1s.append(F1)
                        Accuracies.append(Accuracy)
                    mr[da][a][s].append((np.mean(Accuracies), np.mean(F1s)))
    return mr, nb


def plot(r):
    res = getTestResultsValidation(r)
    mr, nb = getMean(res)

    # algos = ["GMML", "ITML", "LMNN"]
    # algos = ["Euclidean", "GMML", "ITML", "LMNN", "IML"]
    bar_width = 0.75 / len(algos)

    matplotlib.rcParams['font.size'] = 14
    for da in datasets:
        fig = plt.figure(1, figsize=(10, 3))

        perfs = ["Accuracy", "F1-measure"]
        for perfIdx in range(2):
            ax = fig.add_subplot(1, 2, perfIdx+1)
            ax.grid(zorder=0)
            ax.set_title(perfs[perfIdx])
            ax.set_ylabel("")
            ax.set_xlabel("Percentage of positive examples")
            ax.set_ylim([0, 100])
            ax.set_yticks(np.arange(0, 100.1, 10))
            if perfIdx > 0:
                ax.set_yticklabels([])

            index = np.arange(len(nb[da]))
            opacity = 1.0

            ax.set_xticks(index+bar_width*len(algos)/2-bar_width/2)
            labels = [str(int(pct*100)) + "%" for pct in nb[da]]
            ax.set_xticklabels(labels)

            for i, a in enumerate(algos):
                vals = [val[perfIdx]*100 for val in mr[da][a]["test"]]
                ax.bar(index+i*bar_width, vals, bar_width,
                       alpha=opacity,
                       label=a, zorder=2)
            if perfIdx == 0:
                ax.legend(ncol=len(algos), loc=9, bbox_to_anchor=(0.98, 1.32))
        fig.subplots_adjust(wspace=0.04, hspace=0.20)
        plt.savefig("res_" + da + ".pdf", bbox_inches="tight")
        plt.close(fig)


plot(r)
