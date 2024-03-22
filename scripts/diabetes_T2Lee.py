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
from supy.problems.diabetes import (
    DiabetesT2LeeProblem,
)
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

params = {
    "adao.printState": False,
    "populationSize": 20,
    "computationTime": 1000000,
    "minimumEpsilon": 0.4,
    "evaluationBudget": 40,
    "pretrainingBudgetRate": 0.2,
    "deviation": 0.4,
    "nudged.CMagic": 0.1,
    "nudged.K": 0.1,
    "adaoAlgorithm": "3DVAR",
    "cpt.iters": 100,
    "repTime": 1,
    "numberOfSubmodels": 3,
    "pretrainingAlgorithm": assimilation.runADAO,
}

problemParams = {"cpt.timePoints": [0, 100, 200, 300, 400, 500, 600]}

problemType = DiabetesT2LeeProblem

diabetesProblem = problemType((0, 600))
diabetesParams = dict(**problemParams, **params)
diabetesParamsList = [diabetesParams] * 2

expSuite = ExperimentSuite(runners, diabetesProblem, diabetesParamsList)

# %%
# problemType = DiabetesT2AlbersProblem
# math domain error

# problemType = DiabetesT2DrozdovProblem
# ADAO error:
# The "ObservationError" covariance matrix is not positive-definite

# %% [markdown]
# Configure plot contents

# %%
iterPlot = sr.combine(
    sr.generalPlot(runners, 0, "General", ("time", "MTD")),
    # sr.statePlot(runnersWithTitles, [("C", 0), ("P", 1), ("Q", 2), ("QP", 3)],
    #              ("time", "y")),
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
expSuite.run(iterPlot, experimentPlot, suitePlot)
