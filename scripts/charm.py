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

# %%
# !pip install -U pymoo


# %%
def standardGroundTruth():
    numberOfCycles = 1
    data = np.loadtxt(dataFile("tumor/standard.csv"))
    groundTruthObservation = np.tile(data, numberOfCycles)

    def transform(x):
        scalingFactor = 1500
        return (x**3) * np.pi / 6 / scalingFactor

    return list(map(transform, groundTruthObservation))


# %% pycharm={"name": "#%%\n"}
# try:
#   install == False
# except NameError:
# !pip install pyabc
#   from google.colab import drive
#   drive.mount('/gdrive')
# #   !ln -s '/gdrive/My Drive/pyabc' pyabc
# # !ln -s '/Users/macbookpro/Dysk Google/pyabc' pyabc
#   install = True
# %matplotlib inline


# %% pycharm={"name": "#%%\n"}


def plot3D(f, data, plot, cords):
    rows, cols, i = cords
    ax = f.add_subplot(rows, cols, i + 1, projection="3d")
    for item in plot.items:
        d = data[item.key]
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]
        ax.plot(x, y, z, label=item.label)
    ax.legend()


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
therapyMoments = [(0, 1), (10, 5)]
# Lambdap, K, Kqpp, Kpq, Gammap, Gammaq, Deltaqp, KDE
idealX = [
    6.80157379e-01,
    1.60140838e02,
    0.00000001e00,
    4.17370748e-01,
    5.74025981e00,
    1.34300000e00,
    6.78279483e-01,
    9.51318080e-02,
]
# C, P, Q, QP
initState = [0, 4.72795425, 48.51476221, 0]

genericParams = {
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
    "repTime": 2,
    "numberOfSubmodels": 3,
    "pretrainingAlgorithm": assimilation.runADAO,
}

tumorParams = dict(**{"cpt.timePoints": [-10, 0, 10, 20, 30, 40, 50]}, **genericParams)
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

# diabetesProblem = DiabetesT2LeeProblem((0, 600))
# diabetesParams = dict(
#     **{"cpt.timePoints": [0, 100, 200, 300, 400, 500, 600]}, **genericParams
# )
# diabetesParamsList = [diabetesParams] * 2

# expSuite = ExperimentSuite(runners, diabetesProblem, diabetesParamsList)

# expSuite.run(
#     sr.combine(
#         sr.generalPlot(runners, 0, "General", ("time", "MTD")),
#         # sr.statePlot(runnersWithTitles, [("C", 0), ("P", 1), ("Q", 2), ("QP", 3)],
#         #              ("time", "y")),
#         # sr.couplingsPlot(),
#         sr.cptWeightsPlot(),
#         sr.deviationPlot(runners, 0, "Deviation", ("time", "MTD")),
#         legendAttribute={"loc": "best"},
#         vLines=(5, 15),
#     ),
#     sr.combine(
#         sr.averagePlot(runners, 0, "Average"),
#         sr.averageErrorPlot(runners, 0, "AverageError"),
#         axesLabels=("time", "MTD"),
#         legendAttribute={"loc": "best"},
#         vLines=(5, 15),
#     ),
#     sr.combine(
#         sr.suiteErrorPlot(runners, "SuiteAverageError"),
#         sr.suiteErrorDeviationPlot(runners, "SuiteDeviationError"),
#         axesLabels=("Experiments", "Error"),
#         legendAttribute={"loc": "best"},
#     ),
# )


# problem = Lorenz63Problem((0, 20))
# lorenzParams = dict(**{"cpt.timePoints": [0, 10, 20]}, **genericParams)
# paramList = [lorenzParams] * 2

# expSuite = ExperimentSuite(runners, problem, paramList)
# expSuite.run(
#     sr.combine(
#         sr.generalPlot(runners, 0, "General", ("x", "y"), {"loc": "best"}, (5, 15)),
#         sr.statePlot(
#             runnersWithTitles,
#             [("x", 0), ("y", 1), ("z", 2)],
#             ("x", "y"),
#             {"loc": "best"},
#             (5, 15),
#         ),
#     ),
#     sr.combine(
#         sr.averagePlot(runners, 0, "Average"),
#         sr.averageErrorPlot(runners, 0, "AverageError"),
#         axesLabels=("time", "MTD"),
#         legendAttribute={"loc": "best"},
#         vLines=(5, 15),
#     ),
#     sr.combine(
#         sr.suiteErrorPlot(runners, "SuiteAverageError"),
#         sr.suiteErrorDeviationPlot(runners, "SuiteDeviationError"),
#         axesLabels=("Experiments", "Error"),
#         legendAttribute={"loc": "best"},
#     ),
# )

# learningWindows = [(k - 10, k) for k in range(1)]
# experiments = [
#     Experiment(
#         problems.tumor.TumorProblem(
#             modelingWindow, initState, idealX, w, coupledFields
#         ),
#         assimilation.runABC,
#         numberOfSubmodels,
#         params,
#     )
#     for w in learningWindows
# ]
# runTheListOfExperiments(experiments)

# Ponizsze za publikacja: https://www.sciencedirect.com/science/article/pii/S0921800914000615
###################################################################
# HandyModel
# 5.3.2 Unequal Society: Type-L Collapse (Labor Disappears, Nature Recovers)
# alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
# unequalCollapseTypeLIdealX = [
#     1e-2,
#     7e-2,
#     3e-2,
#     3e-2,
#     5e-4,
#     5e-3,
#     1e-2,
#     1e2,
#     100,
#     1.0e-4,
# ]
# xC, xE, y, w
# initState = [0.2, 600, 1e2, 50]
# ###################################################################
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     "repTime": 2,
# }
# coupledFields = [0, 1, 2, 3]
# numberOfSubmodels = 3
# modelingWindow = (0, 1000)
# learningWindow = (0, 1000)
# unequalCollapseTypeLHandyProblem = problems.HandyProblem(
#     modelingWindow,
#     initState,
#     unequalCollapseTypeLIdealX,
#     learningWindow,
#     coupledFields,
# )
# unequalCollapseTypeLHandyExperiment = Experiment(
#     unequalCollapseTypeLHandyProblem,
#     assimilation.runADAO,
#     numberOfSubmodels,
#     params,
# )
# runTheListOfExperiments([unequalCollapseTypeLHandyExperiment] * 2)
##################################################################################

# ###################################################################
# # HandyModel
# # 5.3.2 Unequal Society: Type-L Collapse (Labor Disappears, Nature Recovers)
# # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
# unequalCollapseTypeLIdealX = [
#     1e-2,
#     7e-2,
#     3e-2,
#     3e-2,
#     5e-4,
#     5e-3,
#     1e-2,
#     1e2,
#     100,
#     1.0e-4,
# ]
# # xC, xE, y, w
# initState = [0.2, 600, 1e2, 50]
# ###################################################################
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     "repTime": 2,
# }
# coupledFields = [0, 1, 2, 3]
# numberOfSubmodels = 3
# modelingWindow = (0, 1000)
# learningWindow = (0, 1000)
# unequalCollapseTypeLHandyProblem = HandyProblem(
#     modelingWindow,
#     initState,
#     unequalCollapseTypeLIdealX,
#     learningWindow,
#     coupledFields,
# )
# unequalCollapseTypeLHandyExperiment = Experiment(
#     unequalCollapseTypeLHandyProblem, runADAO, numberOfSubmodels, params
# )
# runTheListOfExperiments([unequalCollapseTypeLHandyExperiment] * 2)
# ##################################################################################

# ###################################################################
# # HandyModel
# # 5.3.3 Unequal Society: Soft Landing to Optimal Equilibrium
# # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
# unequalSoftEquilibriumIdealX = [
#     1e-2,
#     7e-2,
#     6.5e-2,
#     2e-2,
#     5e-4,
#     5e-3,
#     1e-2,
#     1e2,
#     10,
#     6.35e-6,
# ]
# # xC, xE, y, w
# initState = [1.0e4, 3e3, 1e2, 50]
# ###################################################################
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     "repTime": 2,
# }
# coupledFields = [0, 1, 2, 3]
# numberOfSubmodels = 3
# modelingWindow = (0, 1000)
# learningWindow = (0, 1000)
# unequalSoftEquilibriumHandyProblem = HandyProblem(
#     modelingWindow,
#     initState,
#     unequalSoftEquilibriumIdealX,
#     learningWindow,
#     coupledFields,
# )
# unequalSoftEquilibriumHandyExperiment = Experiment(
#     unequalSoftEquilibriumHandyProblem, runADAO, numberOfSubmodels, params
# )
# runTheListOfExperiments([unequalSoftEquilibriumHandyExperiment] * 2)
# ##################################################################################
#
#
#
#
#
# ###################################################################
# # HandyModel
# # 5.3.4 Unequal Society: Oscillatory Approach to Equilibrium
# # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
# unequalOscilatoryIdealX = [
#     1e-2,
#     7e-2,
#     6.5e-2,
#     2e-2,
#     5e-4,
#     5e-3,
#     1e-2,
#     1e2,
#     10,
#     1.3e-5,
# ]
# # xC, xE, y, w
# initState = [1.0e4, 3e3, 1e2, 50]
# ###################################################################
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     "repTime": 2,
# }
# coupledFields = [0, 1, 2, 3]
# numberOfSubmodels = 3
# modelingWindow = (0, 1000)
# learningWindow = (0, 1000)
# unequalOscilatoryHandyProblem = HandyProblem(
#     modelingWindow, initState, unequalOscilatoryIdealX, learningWindow, coupledFields
# )
# unequalOscilatoryHandyExperiment = Experiment(
#     unequalOscilatoryHandyProblem, runADAO, numberOfSubmodels, params
# )
# runTheListOfExperiments([unequalOscilatoryHandyExperiment] * 2)
# ##################################################################################

# problem = HandyProblem((0, 1000))
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
# }

# problem = DiabetesT2Problem((0, 600))
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600],
# }


# problem = DiabetesT2AlbersProblem((0, 600))
# problem = DiabetesT2LeeProblem((0, 600))
# dziala
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600],
# }

# problem = DiabetesT2DrozdovProblem((0, 600))
# nie dziala
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 40,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 100, 200, 300, 400, 500, 600],
# }

# problem = Lorenz63Problem((0, 20))
# params = {
#     "populationSize": 20,
#     "computationTime": 1000000,
#     "minimumEpsilon": 0.4,
#     "evaluationBudget": 5,
#     "deviation": 0.4,
#     "CMagic": 0.1,
#     "K": 0.1,
#     "adaoAlgorithm": "3DVAR",
#     "iters": 100,
#     "timePoints": [0, 10, 20],
# }

# no_param_changes_experiment(repetition)

# %% pycharm={"name": "#%%\n"}
