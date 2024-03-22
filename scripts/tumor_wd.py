# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy

import supy.report as sr
from supy import assimilation
from supy.experiment import ExperimentSuite
from supy.problems.tumor import TumorModelWD, TumorProblemWithTherapy
from supy.strategies import (
    AssimilatedSuperModelRunner,
    CPTSuperModelRunner,
    SingleModelRunner,
    WeightedSuperModelRunner,
)
from supy.utils import dataFile


def readGroundTruthObservation(path):
    csvData = numpy.loadtxt(dataFile(path), skiprows=1, delimiter=",")
    # data = csvData[::1000, 1]
    return csvData


runnersWithTitles = [
    (SingleModelRunner(assimilation.runADAO), "Single"),
    (AssimilatedSuperModelRunner(assimilation.runADAO), "Assimilated"),
    (WeightedSuperModelRunner(assimilation.runADAO), "Weighted"),
    (CPTSuperModelRunner(), "CPT"),
    # (NudgedSuperModelRunner(), "Nudged")
]
runners = [x[0] for x in runnersWithTitles]

#########Proba wywolania modelu WD
# C, P, Q, QP

gt = readGroundTruthObservation("tumor/standard.csv")
coupledFields = [0, 1]
modelingWindow = (0, len(gt) - 1)
learningWindow = (0, 20)
# therapyMoments = [(0,1), (10,5)]
therapyMoments = [(7, 1)]

# , (16600,10),(40900,10),(51500,10),(59100,10)]

# Lambdap, K, Kqpp, Kpq, Gammap, Gammaq, Deltaqp, KDE
# idealX = [
#     6.80157379e-01,
#     1.60140838e02,
#     0.00000001e00,
#     4.17370748e-01,
#     5.74025981e00,
#     1.34300000e00,
#     6.78279483e-01,
#     9.51318080e-02,
# ]


# lambdaP, K, kQ0P, kPH, kPH0, gamma, deltaQH, KDE

idealX = [
    6.80157379e-01,
    750000,
    0.00000001e00,
    4.17370748e-01,
    4.17370748e-01,
    3.74025981e00,
    6.78279483e-01,
    9.51318080e-02,
]

# idealX = [
#     6.80157379e-01,
#     1.60140838e02,
#     0.00000001e00,
#     4.17370748e-01,
#     4.17370748e-01,
#     3.74025981e00,
#     6.78279483e-01,
#     9.51318080e-02,
# ]
# C, P, Q, QP
initState = [0, 4.72795425, 48.51476221, 0]
# initState = [0, 5700, 57000, 0]

# genericParams = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "pretrainingBudgetRate": 0.2,
#     "deviation": 0.4,
#     "nudged.CMagic": 0.1,
#     "nudged.K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "cpt.iters": 100,
#     "repTime": 2,
#     "numberOfSubmodels": 3,
#     "pretrainingAlgorithm": assimilation.runADAO,
# }


genericParams = {
    "populationSize": 20,
    "computationTime": 1000,
    "minimumEpsilon": 0.4,
    "evaluationBudget": 40,
    "pretrainingBudgetRate": 0.2,
    "deviation": 0.6,
    "nudged.CMagic": 0.1,
    "nudged.K": 0.1,
    "adaoAlgorithm": "3DVAR",
    "cpt.iters": 10,
    "repTime": 1,
    "numberOfSubmodels": 3,
    "pretrainingAlgorithm": assimilation.runADAO,
}

tumorParams = dict(**{"cpt.timePoints": [-10, 0, 10, 20, 30, 40, 50]}, **genericParams)
tumorParamsList = [tumorParams] * 2

problemWithTherapy = TumorProblemWithTherapy(
    TumorModelWD,
    modelingWindow,
    initState,
    idealX,
    learningWindow,
    therapyMoments,
    coupledFields,
    gt,
)

expSuite = ExperimentSuite(runners, problemWithTherapy, tumorParamsList)

expSuite.run(
    sr.combine(
        sr.generalPlot(runners, 0, "General", ("time", "MTD")),
        sr.statePlot(
            runnersWithTitles, [("C", 0), ("P", 1), ("Q", 2), ("QP", 3)], ("time", "y")
        ),
        # sr.couplingsPlot(),
        sr.cptWeightsPlot(),
        sr.deviationPlot(runners, 0, "Deviation", ("time", "MTD")),
        legendAttribute={"loc": "best"},
        vLines=(5, 15),
    ),
    sr.combine(
        sr.averagePlot(runners, 0, "Average"),
        sr.averageErrorPlot(runners, 0, "AverageError"),
        axesLabels=("time", "MTD"),
        legendAttribute={"loc": "best"},
        vLines=(5, 15),
    ),
    sr.combine(
        sr.suiteErrorPlot(runners, "SuiteAverageError"),
        sr.suiteErrorDeviationPlot(runners, "SuiteDeviationError"),
        axesLabels=("Experiments", "Error"),
        legendAttribute={"loc": "best"},
    ),
)

# %%
