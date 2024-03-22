import numpy as np
from scipy.integrate import odeint

from .. import trace, utils


def assimilatedSuperModelSolver(
    models, t0, t1, coeffs, coupledFields, groundTruth, K=0
):
    def dt(z, t):
        idx = int(min(t, t1) - t0)
        dzdt = []
        offset = 0
        prev = []
        for m in models:
            zi = z[offset : offset + m.size]
            dzdt.append(m(zi, t))
            offset += m.size
            prev.append(zi)
        nudging = [m.nudging(prev[j], groundTruth[idx]) for j, m in enumerate(models)]
        for i in range(len(models)):
            for cf in coupledFields:
                dzdt[i][cf] += sum(
                    coeffs[(i, j)] * (prev[j][cf] - prev[i][cf])
                    for j in range(len(models))
                    if j != i
                )
                dzdt[i][cf] += K * nudging[i][cf]
        return [x for dz in dzdt for x in dz]

    trace.registerCall()
    n = t1 - t0 + 1
    ts = np.linspace(t0, t1, n)
    s = []
    trajectory = odeint(dt, sum((list(m.initState) for m in models), []), ts)
    s = utils.assembleResults(models, trajectory)
    subOut = {f"S{i}": s[i] for i in range(len(models))}
    return {"data": np.mean(s, 0), "states": trajectory, **subOut}


class AssimilatedSuperModelRunner:
    def __init__(self, assimilationAlgorithm, name=None):
        self.name = name or "assimilated"
        self.assimilationAlgorithm = assimilationAlgorithm

    def __call__(self, experiment, submodelParams):
        problem = experiment.problem

        subModels = [problem.modelWithParams(*x) for x in submodelParams]
        superSolver = problem.createSuperSolver(subModels)

        coeffs = self.assimilationAlgorithm(
            superSolver,
            [0.5] * len(subModels) * (len(subModels) - 1),
            problem.postprocessedGroundTruth,
            problem.postprocess,
            experiment.params,
        )

        return superSolver(*coeffs)

    def outputVars(
        self,
        experiment,
    ):
        outDim = experiment.problem.outDim
        stateDim = experiment.problem.stateDim
        n = experiment.params["numberOfSubmodels"]
        T = utils.getTimeStepCount(experiment.problem)
        return [("data", T, outDim), ("states", T, stateDim * n)] + [
            (f"S{j}", T, outDim) for j in range(n)
        ]
