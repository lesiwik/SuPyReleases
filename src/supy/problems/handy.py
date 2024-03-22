from ..strategies.single import singleModelSolver
from . import base


class HandyModel:
    def __init__(
        self, initState, alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
    ):
        self.initState = initState
        self.alfam = alfam
        self.alfaM = alfaM
        self.betaC = betaC
        self.betaE = betaE
        self.s = s
        self.ro = ro
        self.gamma = gamma
        self.lbd = lbd
        self.kappa = kappa
        self.delta = delta
        self.size = 4

    def __call__(self, z, t):
        xC, xE, y, w = z
        xC = max(1e-2, xC)
        xE = max(1e-2, xE)
        y = max(1e-2, y)
        w = max(1e-2, w)
        wth = self.ro * xC + self.kappa * self.ro * xE
        CC = min(1, w / wth) * self.s * xC
        CE = min(1, w / wth) * self.kappa * self.s * xE
        alfaC = self.alfam + max(0, 1 - CC / (self.s * xC)) * (self.alfaM - self.alfam)
        alfaE = self.alfam + max(0, 1 - CE / (self.s * xE)) * (self.alfaM - self.alfam)

        dxCdt = self.betaC * xC - alfaC * xC
        dxEdt = self.betaE * xE - alfaE * xE
        dydt = self.gamma * y * (self.lbd - y) - self.delta * xC * y
        dwdt = self.delta * xC * y - CC - CE

        return [dxCdt, dxEdt, dydt, dwdt]

    def postprocess(self, z):
        return z

    def nudging(self, z, observation):
        scale = sum(observation) / sum(z)
        return z * (scale - 1)

    def scalarize(self, z):
        return sum(z)

    def withInitState(self, newInitState):
        return HandyModel(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [
            self.alfam,
            self.alfaM,
            self.betaC,
            self.betaE,
            self.s,
            self.ro,
            self.gamma,
            self.lbd,
            self.kappa,
            self.delta,
        ]


class HandyProblem(base.ODESystemProblem):
    def __init__(
        self, learningWindow, initState, idealX, modelingWindow, coupledFields
    ):
        # https://www.sciencedirect.com/science/article/pii/S0921800914000615
        # Egalitarian Society: Egalitarian Society: Soft Landing to Equilibrium
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 0, 6.67e-6]
        # xC, xE, y, w
        # init = [1e2, 0, 1e2, 50]

        # ###################################################################
        # Egalitarian Society: Oscillatory Approach to Equilibrium##########
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 0, 1.67e-5]
        # xC, xE, y, w
        # init = [1e2, 0, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # Egalitarian Society: Cycles of Prosperity, Overshoot, Collapse, and
        # Revival
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 0, 2.67e-5]
        # xC, xE, y, w
        # init = [1e2, 0, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # Egalitarian Society: Irreversible Type-N Collapse (Full Collapse)
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 0, 3.67e-5]
        # xC, xE, y, w
        # init = [1e2, 0, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.2.1 Equitable Society: Soft Landing to Optimal Equilibrium
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 1, 8.33e-6]
        # xC, xE, y, w
        # init = [1e2, 25, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.2.2 Equitable Society: Oscillatory Approach to Equilibrium
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 1, 2.2e-5]
        # xC, xE, y, w
        # init = [1e2, 25, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.2.3 Equitable Society: Cycles of Prosperity, Overshoot, Collapse,
        # and Revival
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 1, 3.0e-5]
        # xC, xE, y, w
        # init = [1e2, 25, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.2.4 Equitable Society: Full Collapse
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 1, 4.33e-5]
        # xC, xE, y, w
        # init = [1e2, 25, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.2.5 Equitable Society: Preventing a Full Collapse by Decreasing Average
        # Depletion per Capita
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 1, 4.33e-5]
        # xC, xE, y, w
        # init = [1e2, 600, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.3.1 Unequal Society: Type-L Collapse (Labor Disappears, Nature
        # Recovers)
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 100, 4.33e-5]
        # xC, xE, y, w
        # init = [1e-3, 600, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.3.2 Unequal Society: Type-L Collapse (Labor Disappears, Nature
        # Recovers)
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 3e-2, 3e-2, 5e-4, 5e-3, 1e-2, 1e2, 100, 1.0e-4]
        # xC, xE, y, w
        # init = [0.2, 600, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.3.3 Unequal Society: Soft Landing to Optimal Equilibrium
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 6.5e-2, 2e-2, 5e-4, 5e-3, 1e-2, 1e2, 10, 6.35e-6]
        # xC, xE, y, w
        # init = [1.0e4, 3e3, 1e2, 50]
        # ###################################################################

        # ###################################################################
        # 5.3.4 Unequal Society: Oscillatory Approach to Equilibrium
        # alfam, alfaM, betaC, betaE, s, ro, gamma, lbd, kappa, delta
        # idealX = [1e-2, 7e-2, 6.5e-2, 2e-2, 5e-4, 5e-3, 1e-2, 1e2, 10, 1.3e-5]
        # xC, xE, y, w
        # init = [1.0e4, 3e3, 1e2, 50]
        # ###################################################################
        tmin, tmax = modelingWindow
        gt = singleModelSolver(HandyModel(initState, *idealX), tmin, tmax)
        groundTruthObservation = gt["data"]
        super().__init__(
            HandyModel,
            initState,
            groundTruthObservation,
            tmin,
            tmax,
            4,
            coupledFields,
            idealX,
            learningWindow,
        )
