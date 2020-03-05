#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import os
import gzip
import sys
from subprocess import call

import numpy as np


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
print(algos)
# ['Euclidean', 'LMNN', 'ITML', 'GMML', 'IMLS', 'IML', 'oEuclidean', 'oLMNN', 'oITML', 'oGMML', 'oIMLS', 'oIML', 'uEuclidean', 'uLMNN', 'uITML', 'uGMML', 'uIMLS', 'uIML']
# algos = ['Euclidean', 'LMNN', 'ITML', 'GMML', 'IMLS', 'IML']
# algos = ['oEuclidean', 'oLMNN', 'oITML', 'oGMML', 'oIMLS', 'oIML']
algos = ['uEuclidean', 'uLMNN', 'uITML', 'uGMML', 'uIMLS', 'uIML']


def getTestResultsValidation(r):
    res = []
    for i in range(len(r)):
        res.append({})
        for da in datasets:
            res[i][da] = {}
            for a in algos:
                sor = []
                for nameP in r[i][da]["valid"][a].keys():
                    perfP = []
                    for perf in r[i][da]["valid"][a][nameP]:
                        tn, fp, fn, tp = perf["test"]
                        F1 = tp/(tp+fn/2+fp/2)
                        perfP.append(F1)
                    sor.append((np.mean(perfP), nameP))
                sor = sorted(sor)
                bestP = sor[-1][1]
                res[i][da][a] = r[i][da]["test"][a][bestP]
    return res


def getMean(r):
    mr = {}
    for da in datasets:
        mr[da] = {}
        for a in algos:
            mr[da][a] = {}
            for s in ["train", "test"]:
                F1s = []
                for i in range(len(r)):
                    tn, fp, fn, tp = r[i][da][a][s]
                    F1 = tp/(tp+fn/2+fp/2)
                    F1s.append(F1)
                mean = np.mean(F1s)
                std = np.std(F1s)
                mr[da][a][s] = ((mean, std))

    all = "Mean"
    mr[all] = {}
    mr[all] = {a:
               {s:
                (np.mean([mr[da][a][s][0] for da in datasets]),
                 np.mean([mr[da][a][s][1] for da in datasets]))
                for s in ["train", "test"]}
               for a in algos}
    return mr


def latex(r):
    if not os.path.exists("latex"):
        os.makedirs("latex")

    f = open('latex/doc.tex', 'w')
    sys.stdout = f
    print(r"\documentclass[a4paper, 12pt]{article}")
    print(r"\usepackage[french]{babel}")
    print(r"\usepackage[T1]{fontenc}")
    print(r"\usepackage{amssymb} ")
    print(r"\usepackage{amsmath}")
    print(r"\usepackage[utf8]{inputenc}")
    print(r"\usepackage{graphicx}")
    print(r"\usepackage{newtxtext}")
    print(r"\usepackage{booktabs}")
    print(r"\usepackage{multirow}")

    print(r"\begin{document}")

    res = getTestResultsValidation(r)
    mr = getMean(res)

    print(r"\begin{table*}")
    print(r"\resizebox{1\textwidth}{!}{\begin{tabular}{l ", end="")
    for a in algos:
        print(" c", end="")
    print("}")
    print(r"\toprule")

    print("{:12}".format("Dataset"), end="")
    ranks = {}
    for a in algos:
        print("&       {:11}".format(a.replace("_", "\\_")), end="")
        ranks[a] = 0
    print(r"\\")
    print(r"\midrule")
    for da in datasets + ["Mean"]:
        if da == "Mean":
            print(r"\midrule")
        print("{:12}".format(da), end="")

        order = list(reversed(
                   np.argsort([mr[da][a]["test"][0] for a in algos])))
        best = algos[order[0]]

        for i, idx in enumerate(order):
            ranks[algos[idx]] += 1+i

        best = algos[np.argmax([mr[da][a]["test"][0] for a in algos])]
        for a in algos:
            b1 = ""
            b2 = ""
            if a == best:
                b1 = r"\textbf{"
                b2 = "}"
            print("&  " + b1 +
                  "{:4.1f}".format(mr[da][a]["test"][0]*100) +
                  b2 + " $\\pm$ {:4.1f}".format(
                                        mr[da][a]["test"][1]*100),
                  end="")
        print(r"\\")
        if da == "Mean":
            print(r"\midrule")
            print("Average Rank", end="")
            for a in algos:
                print(" & {:1.2f}".format(ranks[a]/(len(datasets)+1)), end="")
            print("\\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table*}")

    print(r"\end{document}")
    f.close()

    call(["pdflatex", "-output-directory=latex", "latex/doc.tex"])
    os.remove("latex/doc.aux")
    os.remove("latex/doc.log")


latex(r)
