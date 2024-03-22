import numpy as np

from .. import utils
from ..problems.utils import runWithModifier
from .single import singleModelSolver


def CPTModelSolver(models, t0, t1, groundTruth, params):
    class SuperModel:
        def __init__(self, models, weights, initState):
            self.models = models
            self.weights = weights
            self.initState = initState

        def __call__(self, z, t):
            return sum(
                w * np.array(m(z, t))
                for m, w in zip(self.models, self.weights, strict=True)
            )

        def postprocess(self, z):
            return models[0].postprocess(z)

    stateModifier = params.get("stateModifier") or (lambda _, z: z)
    modificationPoints = params.get("modificationPoints") or []
    weights = [1 / len(models) for _ in models]
    history = [weights]
    ts = sorted(set(params["cpt.timePoints"] + modificationPoints))
    for _ in range(params["cpt.iters"]):
        state = models[0].initState
        hits = [0 for _ in range(len(models) + 1)]
        for a, b in zip(ts, ts[1:], strict=False):
            superModel = SuperModel(models, weights, state)
            allModels = [m.withInitState(state) for m in models] + [superModel]
            results = [singleModelSolver(m, a, b) for m in allModels]
            tmpGT = groundTruth[int(b - t0)]
            idx = np.argmin([np.linalg.norm(tmpGT - r["data"][-1]) for r in results])
            hits[idx] += 1
            state = results[idx]["states"][-1]
            if b in modificationPoints:
                state = stateModifier(b, state)

        weights = [
            (k + hits[-1] * weights[j]) / sum(hits) for j, k in enumerate(hits[:-1])
        ]
        history.append(weights)
        # print("Wagi: {}, suma Wag: {}".format(weights, sum(weights)))

    def solver(initState, a, b):
        return singleModelSolver(SuperModel(models, weights, initState), a, b)

    res = runWithModifier(
        solver, models[0].initState, t0, t1, modificationPoints, stateModifier
    )
    return {**res, "weights": history}


class CPTSuperModelRunner:
    def __init__(self, name=None):
        self.name = name or "cpt"

    def __call__(self, experiment, submodelParams):
        problem = experiment.problem

        subModels = [problem.modelWithParams(*x) for x in submodelParams]

        return problem.cptSolver(subModels, experiment.params)

    def outputVars(self, experiment):
        outDim = experiment.problem.outDim
        stateDim = experiment.problem.stateDim
        n = experiment.params["numberOfSubmodels"]
        T = utils.getTimeStepCount(experiment.problem)
        return [
            ("data", T, outDim),
            ("states", T, stateDim),
            ("weights", experiment.params["cpt.iters"] + 1, n),
        ]
