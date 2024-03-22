from .. import utils


class WeightedSuperModelRunner:
    def __init__(self, assimilationAlgorithm, name=None):
        self.name = name or "weighted"
        self.assimilationAlgorithm = assimilationAlgorithm

    def __call__(self, experiment, submodelParams):
        problem = experiment.problem

        subModels = [problem.modelWithParams(*x) for x in submodelParams]
        solver = problem.createWeightedSolver(subModels)

        weights = self.assimilationAlgorithm(
            solver,
            [0.5] * len(subModels),
            problem.postprocessedGroundTruth,
            problem.postprocess,
            experiment.params,
        )

        return solver(*weights)

    def outputVars(
        self,
        experiment,
    ):
        outDim = experiment.problem.outDim
        stateDim = experiment.problem.stateDim
        n = experiment.params["numberOfSubmodels"]
        T = utils.getTimeStepCount(experiment.problem)
        return [("data", T, outDim), ("states", T, stateDim * n)]
