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

# %% jupyter={"source_hidden": true}
import random

import numpy as np

import supy.report as sr
from supy import assimilation
from supy.experiment import ExperimentSuite
from supy.problems.tumor import TumorModel, TumorProblemWithTherapy
from supy.strategies import (
    AssimilatedSuperModelRunner,
    CPTSuperModelRunner,
    SingleModelRunner,
    WeightedSuperModelRunner,
)
from supy.utils import dataFile

# %% [markdown]
# Ground truth observation needs to be transformed from radius to volume


# %%
def standardGroundTruth():
    numberOfCycles = 1
    data = np.loadtxt(dataFile("tumor/standard.csv"))
    groundTruthObservation = np.tile(data, numberOfCycles)

    def transform(x):
        scalingFactor = 1500
        return (x**3) * np.pi / 6 / scalingFactor

    return list(map(transform, groundTruthObservation))


# %%
runnersWithTitles = [
    (SingleModelRunner(assimilation.runADAO), "Single"),
    (AssimilatedSuperModelRunner(assimilation.runADAO), "Assimilated"),
    (WeightedSuperModelRunner(assimilation.runADAO), "Weighted"),
    (CPTSuperModelRunner(), "CPT"),
    # (NudgedSuperModelRunner(), "Nudged")
]
runners = [x[0] for x in runnersWithTitles]

# C, P, Q, QP
coupledFields = [0, 1]
modelingWindow = (-10, 50)
learningWindow = (10, 20)
therapyMoments = [(0, 1)]

referenceSolution = {
    "Lambdap": 6.80157379e-01,
    "K": 1.60140838e02,
    "Kqpp": 0.00000001e00,
    "Kpq": 4.17370748e-01,
    "Gammap": 5.74025981e00,
    "Gammaq": 1.34300000e00,
    "Deltaqp": 6.78279483e-01,
    "KDE": 9.51318080e-02,
}

idealX = list(referenceSolution.values())

# C, P, Q, QP
initState = [0, 4.72795425, 48.51476221, 0]

genericParams = {
    "populationSize": 20,
    "computationTime": 1000,
    "minimumEpsilon": 0.4,
    "evaluationBudget": 40,
    "pretrainingBudgetRate": 0.2,
    "deviation": 0.4,
    "nudged.CMagic": 0.1,
    "nudged.K": 0.1,
    "adaoAlgorithm": "3DVAR",
    "cpt.iters": 10,
    "repTime": 1,
    "numberOfSubmodels": 3,
    "pretrainingAlgorithm": assimilation.runADAO,
}

problemParams = {
    "cpt.timePoints": [-10, 0, 10, 20, 30, 40, 50],
}

tumorParams = dict(**problemParams, **genericParams)
tumorParamsList = [tumorParams] * 2

problemWithTherapy = TumorProblemWithTherapy(
    TumorModel,
    modelingWindow,
    initState,
    idealX,
    learningWindow,
    therapyMoments,
    coupledFields,
    standardGroundTruth(),
)

expSuite = ExperimentSuite(runners, problemWithTherapy, tumorParamsList)

# %% [markdown]
# Configure plot contents

# %%
iterPlot = sr.combine(
    sr.generalPlot(runners, 0, "General", ("time", "MTD")),
    sr.statePlot(
        runnersWithTitles, [("C", 0), ("P", 1), ("Q", 2), ("QP", 3)], ("time", "y")
    ),
    # sr.couplingsPlot(),
    sr.cptWeightsPlot(),
    sr.deviationPlot(runners, 0, "Deviation", ("time", "MTD")),
    legendAttribute={"loc": "best"},
    vLines=(5, 15),
)

experimentPlot = sr.combine(
    sr.averagePlot(runners, 0, "Average"),
    sr.averageErrorPlot(runners, 0, "AverageError"),
    axesLabels=("time", "MTD"),
    legendAttribute={"loc": "best"},
    vLines=(5, 15),
)

suitePlot = sr.combine(
    sr.suiteErrorPlot(runners, "SuiteAverageError"),
    sr.suiteErrorDeviationPlot(runners, "SuiteDeviationError"),
    axesLabels=("Experiments", "Error"),
    legendAttribute={"loc": "best"},
)

# %% [markdown]
# Execute the model and produce plots

# %%
seed = 3
random.seed(seed)
np.random.seed(seed)

expSuite.run(iterPlot, experimentPlot, suitePlot)
