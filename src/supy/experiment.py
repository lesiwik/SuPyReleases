import copy

import numpy as np

from . import plot, utils


def fillProblemDetails(data, problem):
    data["gt"] = problem.groundTruth
    data["t"] = np.linspace(problem.t0, problem.t1, problem.t1 - problem.t0 + 1)


def runExperimentSuite(experiments, tasks, afterIter, afterExperiment):
    results = {}
    for e in experiments:
        res = runExperiment(e, tasks, lambda n, data, e=e: afterIter(n, data, e))
        for name, val in res.items():
            vals = results.setdefault(name, [])
            vals.append(val)
        fillProblemDetails(res, e.problem)
        afterExperiment(res, e)
    return results


def runExperiment(experiment, tasks, afterIter):
    repTime = experiment.params["repTime"]
    numberOfSubmodels = experiment.params["numberOfSubmodels"]
    problem = experiment.problem
    tasks = tasks + [PretrainedModelRunner(i) for i in range(numberOfSubmodels)]
    outVars = sum(
        (
            [(f"{t.name}.{n}", s, k) for n, s, k in t.outputVars(experiment)]
            for t in tasks
        ),
        [],
    )

    results = {name: utils.Results(repTime, size, dim) for name, size, dim in outVars}
    scale = experiment.params["pretrainingBudgetRate"]
    pretrainingParams = utils.scaleBudget(experiment.params, scale)
    assimilationAlg = experiment.params["pretrainingAlgorithm"]
    experiment = copy.copy(experiment)
    experiment.params = utils.scaleBudget(experiment.params, 1 - scale)
    for i in range(repTime):
        submodelParams = [
            assimilationAlg(
                problem.solver,
                problem.referenceModelParams,
                problem.postprocessedGroundTruth,
                problem.postprocess,
                pretrainingParams,
            )
            for _ in range(numberOfSubmodels)
        ]

        for task in tasks:
            out = task(experiment, submodelParams)
            for k, val in out.items():
                key = task.name + "." + k
                results[key].addResults(val)
        data = {key: r.getLastResult() for key, r in results.items()}
        fillProblemDetails(data, experiment.problem)
        afterIter(i, data)
    return results


def printIterSummary(data, iteration, items):
    print(f"Blad w iteracji {iteration + 1}")
    print("-" * 40)
    for key in items:
        result = data[key + ".data"]
        r = utils.Results(1, *result.shape)
        r.addResults(result)
        text = "{:20} {:10.5f} {:10.5f}%"
        # text = 'Blad {} w: {} powtorzeniu eksperymentu , wynosi: {} ({}%))'
        print(
            text.format(
                key, r.totalError(0, data["gt"]), r.totalProcError(0, data["gt"])
            )
        )


def printExperimentSummary(data, items):
    print("-" * 40)
    fmt = "{:20} {:10#} {:10#} {:10#} {:10#} {:10#} {:10#}"
    header = fmt.replace("#", "").format(
        "Metoda", "Sr. Blad", "Std", "Blad", "Std", "Blad Proc", "Std Proc"
    )
    print(header)
    for key in items:
        result = data[key + ".data"]
        text = fmt.replace("#", ".5f").format(
            key,
            np.mean(result.totalErrors),
            np.std(result.totalErrors),
            np.mean(result.avgError),
            np.std(result.avgError),
            np.mean(result.totalProcErrors),
            np.mean(result.avgProcError),
        )
        print(text)


def printSuiteSummary(data, items):
    for key in items:
        meanResult = data[key + ".mean"]
        stdResult = data[key + ".std"]
        print("Sredni blad " + key)
        print(meanResult)
        print("Odchylenie bledu " + key)
        print(stdResult)


class ExperimentSuite:
    def __init__(self, runnerList, problem, paramList):
        self.runnerList = runnerList
        self.problem = problem
        self.paramList = paramList

    def run(self, iterPlot, expPlot, suitePlot):
        methods = [r.name for r in self.runnerList]

        def afterIter(n, data, experiment):
            printIterSummary(data, n, methods)
            plotDesc = iterPlot(experiment.params)
            plot.plotFigure(plotDesc, data, 2, (15, 8))

        def afterExperiment(data, experiment):
            for m in methods:
                res = data[m + ".data"]
                res.compute(self.problem.groundTruth)
            printExperimentSummary(data, methods)
            plotDesc = expPlot(experiment.params)
            plot.plotFigure(plotDesc, data, 2, (15, 8))

        r = runExperimentSuite(
            [Experiment(self.problem, p) for p in self.paramList],
            self.runnerList,
            afterIter,
            afterExperiment,
        )
        meanData = {
            m + ".mean": np.array([np.mean(x.totalProcErrors) for x in r[m + ".data"]])
            for m in methods
        }
        stdData = {
            m + ".std": np.array([np.std(x.totalProcErrors) for x in r[m + ".data"]])
            for m in methods
        }
        data = dict(**meanData, **stdData)
        printSuiteSummary(data, methods)
        plot.plotFigure(suitePlot(self.paramList), data, 2, (15, 8))


class Experiment:
    def __init__(self, problem, params):
        self.problem = problem
        self.params = params


class PretrainedModelRunner:
    def __init__(self, idx):
        self.name = f"submodel{idx}"
        self.idx = idx

    def __call__(self, experiment, submodelParams):
        problem = experiment.problem
        vars = submodelParams[self.idx]
        return problem.solver(*vars)

    def outputVars(self, experiment):
        T = utils.getTimeStepCount(experiment.problem)
        return [
            ("data", T, experiment.problem.outDim),
            ("states", T, experiment.problem.stateDim),
        ]
