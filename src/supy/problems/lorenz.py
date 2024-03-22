from ..strategies.single import singleModelSolver
from . import base


class Lorenz63Model:
    def __init__(self, initState, sigma, ro, beta):
        self.initState = initState
        self.sigma = sigma
        self.ro = ro
        self.beta = beta
        self.size = 3
        self.outDim = 3

    def __call__(self, zin, t):
        x, y, z = zin
        dxdt = self.sigma * (y - x)
        dydt = x * (self.ro - z) - y
        dzdt = x * y - self.beta * z
        out = [dxdt, dydt, dzdt]
        return out

    def postprocess(self, z):
        return (z[0], z[1], z[2])

    def scalarize(self, z):
        return sum(z)

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def withInitState(self, newInitState):
        return Lorenz63Model(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [self.sigma, self.ro, self.beta]


class Lorenz63Problem(base.ODESystemProblem):
    def __init__(self, learningWindow):
        idealX = [10, 28, 8 / 3]
        init = [10, 10, 10]
        tmin = 0
        tmax = 20
        gt = singleModelSolver(Lorenz63Model(init, *idealX), tmin, tmax)
        groundTruthObservation = gt["data"]
        super().__init__(
            Lorenz63Model,
            init,
            groundTruthObservation,
            tmin,
            tmax,
            3,
            [0, 1, 2],
            idealX,
            learningWindow,
        )
