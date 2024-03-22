import math

import numpy as np
from scipy.integrate import odeint

from .. import trace, utils


def linearCoeffIndex(i, j, modelNumber):
    return i * (modelNumber - 1) + (j if j < i else j - 1)


def coeffsToArray(coeffs, models):
    n = len(models)
    array = [None] * n * (n - 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            array[linearCoeffIndex(i, j, n)] = coeffs[(i, j)]
    return array


def nudgedModelSolver(models, t0, t1, coeffs, coupledFields, groundTruth, params):
    def Cidx(i, j):
        return linearCoeffIndex(i, j, len(models))

    def dt(z, t):
        dzdt = []
        offset = 0
        prev = []
        idx = int(min(t, t1) - t0)
        cMagic = params["nudged.CMagic"]
        K = params["nudged.K"]
        for m in models:
            zi = z[offset : offset + m.size]
            # print("zi: {}".format(zi))
            dzdt.append(m(zi, t))
            offset += m.size
            prev.append(zi)

        def C(i, j):
            return z[offset + Cidx(i, j)]

        nudging = [m.nudging(prev[j], groundTruth[idx]) for j, m in enumerate(models)]
        for i in range(len(models)):
            for cf in coupledFields:
                dzdt[i][cf] += sum(
                    C(i, j) * (prev[j][cf] - prev[i][cf])
                    for j in range(len(models))
                    if j != i
                )
                dzdt[i][cf] += K * nudging[i][cf]

        def state(i):
            return models[i].postprocess(prev[i])

        dzdt = [x for dz in dzdt for x in dz]
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    continue
                si = models[i].scalarize(state(i))
                sj = models[j].scalarize(state(j))
                gt = models[i].scalarize(groundTruth[idx])

                dCijdt = cMagic * math.atan((gt - si) * (si - sj))
                dzdt.append(dCijdt)
        return dzdt

    trace.registerCall()
    n = t1 - t0 + 1
    ts = np.linspace(t0, t1, n)
    s = []
    init = sum((list(m.initState) for m in models), [])
    initCoeffs = coeffsToArray(coeffs, models)
    trajectory = odeint(dt, np.array(init + initCoeffs), ts)
    # res = solve_ivp(
    #     dt, (t0, t1), np.array(init + initCoeffs), t_eval=ts, method="RK45"
    # )
    # trajectory = np.transpose(res.y)
    s = utils.assembleResults(models, trajectory)

    indices = range(len(models))
    offset = sum(m.size for m in models)
    cHistory = {
        f"C{i}{j}": trajectory[:, offset + Cidx(i, j)]
        for i in indices
        for j in indices
        if i != j
    }
    subOut = {f"S{i}": s[i] for i in range(len(models))}
    return {"data": np.mean(s, 0), "states": trajectory, **subOut, **cHistory}


class NudgedSuperModelRunner:
    def __init__(self, name=None):
        self.name = name or "nudged"

    def getCoeffNames(self, numberOfSubmodels):
        return [f"C{i}{j}" for i, j in self.getIndices(numberOfSubmodels)]

    def getIndices(self, numberOfSubmodels):
        models = range(numberOfSubmodels)
        return [(i, j) for i in models for j in models if i != j]

    def __call__(self, experiment, submodelParams):
        problem = experiment.problem
        subModels = [problem.modelWithParams(*x) for x in submodelParams]
        stage1 = problem.classicSolver(subModels, experiment.params)
        numberOfSubmodels = experiment.params["numberOfSubmodels"]
        indices = self.getIndices(numberOfSubmodels)
        coeffNames = self.getCoeffNames(numberOfSubmodels)
        C = {
            idx: stage1[name][-1] for idx, name in zip(indices, coeffNames, strict=True)
        }
        coeffs = coeffsToArray(C, subModels)
        solver = problem.createSuperSolver(subModels, experiment.params["nudged.K"])
        stage2 = solver(*coeffs)
        out = {s: stage1[s] for s in coeffNames}
        out["firstStage.data"] = stage1["data"]
        out["firstStage.states"] = stage1["states"]
        out["data"] = stage2["data"]
        out["states"] = stage2["states"]
        return out

    def outputVars(self, experiment):
        outDim = experiment.problem.outDim
        stateDim = experiment.problem.stateDim
        n = experiment.params["numberOfSubmodels"]
        T = utils.getTimeStepCount(experiment.problem)
        coeffNames = self.getCoeffNames(n)

        return [
            ("data", T, outDim),
            ("states", T, n * stateDim),
            ("firstStage.data", T, outDim),
            ("firstStage.states", T, n * stateDim + len(coeffNames)),
        ] + [(s, T, 1) for s in coeffNames]
