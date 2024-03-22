# from .. import solvers
import numpy as np

from ..strategies.assimilated import assimilatedSuperModelSolver
from ..strategies.cpt import CPTModelSolver
from ..strategies.nudged import linearCoeffIndex, nudgedModelSolver
from ..strategies.single import singleModelSolver
from .utils import runWithModifier


class ODESystemProblem:
    def __init__(
        self,
        model,
        initState,
        groundTruthObservation,
        t0,
        t1,
        outDim,
        coupledFields,
        referenceModelParams,
        learningWindow,
    ):
        self.initState = initState
        self.groundTruth = groundTruthObservation
        self.t0 = t0
        self.t1 = t1
        self.coupledFields = coupledFields
        self.referenceModelParams = referenceModelParams
        self.learningWindow = learningWindow
        self.postprocessedGroundTruth = self.postprocess(self.groundTruth)
        self.outDim = outDim
        self.model = model

    @property
    def stateDim(self):
        return len(self.initState)

    def solver(self, *args):
        model = self.model(self.initState, *args)
        return singleModelSolver(model, self.t0, self.t1)

    def createSuperSolver(self, subModels, K=0):
        def eval(*args):
            n = len(subModels)
            coeffs = {
                (i, j): args[linearCoeffIndex(i, j, n)]
                for i in range(n)
                for j in range(n)
                if i != j
            }
            return assimilatedSuperModelSolver(
                subModels,
                self.t0,
                self.t1,
                coeffs,
                self.coupledFields,
                self.groundTruth,
                K,
            )

        return eval

    def createWeightedSolver(self, subModels):
        def eval(*args):
            res = [singleModelSolver(m, self.t0, self.t1) for m in subModels]
            val = sum(np.array(s["data"]) * w for s, w in zip(res, args, strict=True))
            return {"data": val}

        return eval

    def classicSolver(self, subModels, params):
        coeffs = {
            (i, j): 0.5
            for i in range(len(subModels))
            for j in range(len(subModels))
            if i != j
        }
        return nudgedModelSolver(
            subModels,
            self.t0,
            self.t1,
            coeffs,
            self.coupledFields,
            self.groundTruth,
            params,
        )

    def cptSolver(self, subModels, params):
        return CPTModelSolver(subModels, self.t0, self.t1, self.groundTruth, params)

    def postprocess(self, x):
        a, b = self.learningWindow
        return x[a : b + 1]

    def modelWithParams(self, *args):
        return self.model(self.initState, *args)


class GeneralizedODESystemProblem(ODESystemProblem):
    def __init__(self, *args):
        super().__init__(*args)

    def run(self, solver, initState, modifier):
        return runWithModifier(
            solver, initState, self.t0, self.t1, self.modificationPoints(), modifier
        )

    def solver(self, *args):
        def solver(initState, t0, t1):
            model = self.model(initState, *args)
            return singleModelSolver(model, t0, t1)

        return self.run(solver, self.initState, self.modifyState)

    def modifyExtendedState(self, n):
        def modify(t, state):
            dim = self.stateDim
            for i in range(n):
                state[dim * i : dim * (i + 1)] = self.modifyState(
                    t, state[dim * i : dim * (i + 1)]
                )
            return state

        return modify

    def createSuperSolver(self, subModels, K=0):
        dim = self.stateDim

        def eval(*args):
            n = len(subModels)
            coeffs = {
                (i, j): args[linearCoeffIndex(i, j, n)]
                for i in range(n)
                for j in range(n)
                if i != j
            }

            def solver(initState, t0, t1):
                submodelsWithAlignedInitstates = [
                    self.model(initState[dim * i : dim * (i + 1)], *m.getModelParams())
                    for i, m in enumerate(subModels)
                ]
                return assimilatedSuperModelSolver(
                    submodelsWithAlignedInitstates,
                    t0,
                    t1,
                    coeffs,
                    self.coupledFields,
                    self.groundTruth[t0 - self.t0 :],
                    K,
                )

            return self.run(solver, self.initState * n, self.modifyExtendedState(n))

        return eval

    def createWeightedSolver(self, subModels):
        dim = self.stateDim

        def eval(*args):
            n = len(subModels)

            def solver(initState, t0, t1):
                submodelsWithAlignedInitstates = [
                    self.model(initState[dim * i : dim * (i + 1)], *m.getModelParams())
                    for i, m in enumerate(subModels)
                ]
                res = [
                    singleModelSolver(m, t0, t1) for m in submodelsWithAlignedInitstates
                ]
                data = sum(
                    np.array(s["data"]) * w for s, w in zip(res, args, strict=True)
                )
                states = np.concatenate([r["states"] for r in res], axis=1)
                return {"data": data, "states": states}

            return self.run(solver, self.initState * n, self.modifyExtendedState(n))

        return eval

    def classicSolver(self, subModels, params):
        n = len(subModels)
        dim = self.stateDim

        def solver(initState, t0, t1):
            submodelsWithAlignedInitstates = [
                self.model(initState[dim * i : dim * (i + 1)], *m.getModelParams())
                for i, m in enumerate(subModels)
            ]

            def Cidx(i, j):
                return linearCoeffIndex(i, j, n)

            coeffs = {
                (i, j): initState[dim * n + Cidx(i, j)]
                for i in range(n)
                for j in range(n)
                if i != j
            }

            res = nudgedModelSolver(
                submodelsWithAlignedInitstates,
                t0,
                t1,
                coeffs,
                self.coupledFields,
                self.groundTruth[t0 - self.t0 :],
                params,
            )
            return res

        initState = self.initState * n + [0.5] * n * (n - 1)
        return self.run(solver, initState, self.modifyExtendedState(n))

    def cptSolver(self, subModels, params):
        return CPTModelSolver(
            subModels,
            self.t0,
            self.t1,
            self.groundTruth,
            {
                **params,
                "stateModifier": self.modifyState,
                "modificationPoints": self.modificationPoints(),
            },
        )
