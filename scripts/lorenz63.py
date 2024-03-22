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
import supy.report as sr
from supy import assimilation
from supy.experiment import ExperimentSuite
from supy.problems.lorenz import Lorenz63Problem
from supy.strategies import (
    AssimilatedSuperModelRunner,
    CPTSuperModelRunner,
    SingleModelRunner,
    WeightedSuperModelRunner,
)

# %%
runnersWithTitles = [
    (SingleModelRunner(assimilation.runADAO), "Single"),
    (AssimilatedSuperModelRunner(assimilation.runADAO), "Assimilated"),
    (WeightedSuperModelRunner(assimilation.runADAO), "Weighted"),
    (CPTSuperModelRunner(), "CPT"),
    # (NudgedSuperModelRunner(), "Nudged")
]
runners = [x[0] for x in runnersWithTitles]

genericParams = {
    "populationSize": 20,
    "computationTime": 1000000,
    "minimumEpsilon": 0.4,
    "evaluationBudget": 40,
    "pretrainingBudgetRate": 0.2,
    "deviation": 0.1,
    "nudged.CMagic": 0.1,
    "nudged.K": 0.1,
    "adaoAlgorithm": "3DVAR",
    "cpt.iters": 10,
    "repTime": 1,
    "numberOfSubmodels": 3,
    "pretrainingAlgorithm": assimilation.runADAO,
}

problemParams = {"cpt.timePoints": [0, 10, 20]}

problem = Lorenz63Problem((0, 20))
lorenzParams = dict(**problemParams, **genericParams)
paramList = [lorenzParams] * 2

expSuite = ExperimentSuite(runners, problem, paramList)

# %% [markdown]
# Configure plot contents

# %%
iterPlot = sr.combine(
    sr.generalPlot(runners, 0, "General", ("x", "y"), {"loc": "best"}, (5, 15)),
    sr.statePlot(
        runnersWithTitles,
        [("x", 0), ("y", 1), ("z", 2)],
        ("x", "y"),
        {"loc": "best"},
        (5, 15),
    ),
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
expSuite.run(iterPlot, experimentPlot, suitePlot)
