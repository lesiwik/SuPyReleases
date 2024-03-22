import math
from math import exp, log

from .. import utils
from ..strategies.single import singleModelSolver
from . import base


class DiabetesT2Model:
    # Model z SuperThesis (z tekstu ksiazki)
    # https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1236&context=hmc_theses
    def __init__(self, initState, E, Vi, ti):
        self.initState = initState
        self.E = E
        self.Vi = Vi
        self.ti = ti
        self.Rm = 209
        self.a1 = 6.67
        self.alfa = 7.5
        self.beta = 1.77
        self.Vp = 3
        self.Vg = 10
        self.tp = 6
        self.td = 36
        self.C1 = 300
        self.C2 = 144
        self.C3 = 100
        self.Ub = 72
        self.U0 = 4
        self.Um = 92
        self.Rg = 180
        self.Ig = 216
        self.size = 6

    def __call__(self, z, t):
        Ip, Ii, G, h1, h2, h3 = z
        dIpdt = (
            self.Rm / (1 - exp((-G / (self.Vg * self.C1)) + self.a1))
            - self.E * ((Ip / self.Vp) - (Ii / self.Vi))
            - (Ip / self.tp)
        )

        dIidt = self.E * ((Ip / self.Vp) - (Ii / self.Vi)) - (Ii / self.ti)

        ti_ = Ii * ((1 / self.Vi) + (1 / (self.E * self.ti)))
        dGdt = (
            self.Rg / (1 + exp((0.29 * h3) / (self.Vp - 7.5)))
            + self.Ig
            - self.Ub * (1 - exp(-G / (self.C2 * self.Vg)))
            - 90 / (1 + exp(-1.772 * log(ti_) + 7.76))
            + 4
        )

        dh1dt = (3 * (Ip - h1)) / self.td
        dh2dt = (3 * (h1 - h2)) / self.td
        dh3dt = (3 * (h2 - h3)) / self.td
        return [dIpdt, dIidt, dGdt, dh1dt, dh2dt, dh3dt]

    def postprocess(self, z):
        return z[2]

    def scalarize(self, z):
        return z

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def withInitState(self, newInitState):
        return DiabetesT2Model(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [self.E, self.Vi, self.ti]


class DiabetesT2LeeModel:
    # Model z SuperThesis ale z githuba
    # https://github.com/CassidyLe98/Thesis_KalmanFilters/blob/master/Extended_KFs/Albers/AlbersODE.m
    def __init__(self, initState, E, Vi, ti):
        self.initState = initState
        self.E = E
        self.Vi = Vi
        self.ti = ti
        self.a1 = 6.67
        self.alfa = 7.5
        self.beta = 1.77
        self.Vp = 3
        self.Vg = 10
        self.tp = 6
        self.td = 36
        self.Rm = 209
        self.C1 = 300
        self.C2 = 144
        self.C3 = 100
        self.C4 = 80
        self.C5 = 26
        self.Ub = 72
        self.U0 = 4
        self.Um = 94
        self.Rg = 180
        self.Ig = 216
        self.size = 6

    def __call__(self, z, t):
        Ip, Ii, G, h1, h2, h3 = z
        dIpdt = (
            self.Rm / (1 + exp(-G / (self.Vg * self.C1) + self.a1))
            - self.E * ((Ip / self.Vp) - (Ii / self.Vi))
            - (Ip / self.tp)
        )
        dIidt = self.E * ((Ip / self.Vp) - (Ii / self.Vi)) - (Ii / self.ti)

        dGdt = (
            self.Rg / (1 + exp(0.29 * h3 / self.Vp - 7.5))
            + self.Ig
            - self.Ub * (1 - exp(-G / (self.C2 * self.Vg)))
            - (0.01 * G / self.Vg)
            * (
                90
                / (
                    1
                    + exp(
                        -1.772 * log(Ii * (1 / self.Vi + 1 / (self.E * self.ti))) + 7.76
                    )
                )
                + 4
            )
        )
        dh1dt = (3 * (Ip - h1)) / self.td
        dh2dt = (3 * (h1 - h2)) / self.td
        dh3dt = (3 * (h2 - h3)) / self.td
        return [dIpdt, dIidt, dGdt, dh1dt, dh2dt, dh3dt]

    def postprocess(self, z):
        return z[2]

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def scalarize(self, z):
        return z

    def withInitState(self, newInitState):
        return DiabetesT2LeeModel(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [self.E, self.Vi, self.ti]


class DiabetesT2AlbersModel:
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0048058
    # https://doi.org/10.1371/journal.pone.0048058
    def __init__(self, initState, E, Vi, ti):
        self.initState = initState
        self.E = E
        self.Vi = Vi
        self.ti = ti
        self.Rm = 209
        self.a1 = 6.67
        self.alfa = 7.5
        self.beta = 1.77
        self.Vp = 3
        self.Vg = 10
        self.tp = 6
        self.td = 12
        self.C1 = 300
        self.C2 = 144
        self.C3 = 100
        self.C4 = 80
        self.C5 = 26
        self.Ub = 72
        self.U0 = 4
        self.Um = 94
        self.Rg = 180
        self.Ig = 216
        self.size = 6

    def __call__(self, z, t):
        Ip, Ii, G, h1, h2, h3 = z
        f1 = self.Rm / (1 - exp(-G / (self.Vg * self.C1) + self.a1))
        dIpdt = f1 - self.E * ((Ip / self.Vp) - (Ii / self.Vi)) - (Ip / self.tp)
        dIidt = self.E * ((Ip / self.Vp) - (Ii / self.Vi)) - Ii / self.ti

        f2 = self.Ub * (1 - exp(-G / (self.C2 * self.Vg)))
        f4 = self.Rg / (1 + exp(self.alfa * ((h3 / (self.C5 * self.Vp)) - 1)))

        k = (1 / self.C4) * (1 / self.Vi - 1 / (self.E * self.ti))
        beta_ = 1 + math.pow(k * Ii, -self.beta)
        f3 = (1 / (self.C3 * self.Vg)) * (self.U0 + (self.Um - self.U0) / beta_)
        dGdt = f4 + self.Ig - f2 - f3 * G

        dh1dt = (Ip - h1) / self.td
        dh2dt = (h1 - h2) / self.td
        dh3dt = (h2 - h3) / self.td
        return [dIpdt, dIidt, dGdt, dh1dt, dh2dt, dh3dt]

    def postprocess(self, z):
        return z[2]

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def scalarize(self, z):
        return z

    def withInitState(self, newInitState):
        return DiabetesT2LeeModel(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [self.E, self.Vi, self.ti]


class DiabetesT2DrozdovModel:
    # https://doi.org/10.1016/0895-7177(95)00108-E
    def __init__(self, initState, E, V2, T2):
        self.initState = initState
        self.E = E
        self.T1 = 3
        self.T2 = T2
        self.a = 5.21
        self.b = -0.03
        self.p0 = 72
        self.size = 3
        self.V1 = 3
        self.V2 = V2
        self.V3 = 10
        self.T = 45
        self.L = 100
        self.size = 3

    def __call__(self, zin, t):
        x, y, z = zin
        cx = x / self.V1

        cy = y / self.V2
        # cz = 0.1 * z / self.V3
        f1 = 210 / (1 + exp(self.a + (self.b * (x / self.V1))))
        # q = self.E * (self.cx - self.cy)
        # r1 = x / self.T1
        # r2 = y / self.T2
        f2 = (
            9 / (1 + exp(7.76 - 1.772 * log(1 + self.V2 / (self.E * self.T2)) * cy))
        ) + 0.4
        f3 = 160 / (1 + exp((0.29 / cx) - 7.5))
        dxdt = (
            f1 * (0.1 * z / self.V3)
            - ((self.E / self.V1) + (1 / self.T1)) * x
            + (self.E / self.V2) * y
        )
        dydt = (self.E / self.V1) * x - ((self.E / self.V2) + (1 / self.T2)) * y
        dzdt = (
            f3 * (x / self.V1)
            - (0.1 * z / self.V3) * f2 * (y / self.V2)
            + (self.L - self.p0)
        )
        return [dxdt, dydt, dzdt]

    def postprocess(self, z):
        return z[2]

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def scalarize(self, z):
        return z

    def withInitState(self, newInitState):
        return DiabetesT2LeeModel(newInitState, *self.getModelParams())

    def getModelParams(self):
        return [self.E, self.T2, self.V2]


class DiabetesT2AlbersProblem(base.ODESystemProblem):
    def __init__(self, learningWindow):
        idealX = [0.2, 11, 100]
        init = [200, 200, 12000, 0.1, 0.2, 0.1]
        gt = singleModelSolver(DiabetesT2AlbersModel(init, *idealX), 0, 1440)
        groundTruthObservation = gt["data"]
        super().__init__(
            init,
            utils.ensure2D(groundTruthObservation),
            0,
            1440,
            1,
            [0, 1, 2, 3, 4, 5],
            idealX,
            learningWindow,
        )

    def model(self, *args):
        return DiabetesT2AlbersModel(self.initState, *args)


class DiabetesT2LeeProblem(base.ODESystemProblem):
    # Model z SuperThesis ale z githuba
    # https://github.com/CassidyLe98/Thesis_KalmanFilters/blob/master/Extended_KFs/Albers/AlbersODE.m
    def __init__(self, learningWindow):
        idealX = [0.2, 11, 100]
        init = [200, 200, 12000, 0.1, 0.2, 0.1]
        gt = singleModelSolver(DiabetesT2LeeModel(init, *idealX), 0, 600)
        groundTruthObservation = gt["data"]
        super().__init__(
            DiabetesT2LeeModel,
            init,
            utils.ensure2D(groundTruthObservation),
            0,
            600,
            1,
            [0, 1, 2, 3, 4, 5],
            idealX,
            learningWindow,
        )


class DiabetesT2Problem(base.ODESystemProblem):
    # Model z SuperThesis (z tekstu ksiazki)
    # https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1236&context=hmc_theses
    def __init__(self, learningWindow):
        idealX = [0.2, 11, 100]
        init = [200, 200, 12000, 0.1, 0.2, 0.1]
        gt = singleModelSolver(DiabetesT2Model(init, *idealX), 0, 600)
        groundTruthObservation = gt["data"]
        super().__init__(
            DiabetesT2Model,
            init,
            utils.ensure2D(groundTruthObservation),
            0,
            600,
            1,
            [0, 1, 2, 3, 4, 5],
            idealX,
            learningWindow,
        )


class DiabetesT2DrozdovProblem(base.ODESystemProblem):
    # # https://doi.org/10.1016/0895-7177(95)00108-E
    def __init__(self, learningWindow):
        idealX = [0.2, 11, 100]
        init = [30, 20, 120]
        gt = singleModelSolver(DiabetesT2DrozdovModel(init, *idealX), 0, 600)
        groundTruthObservation = gt["data"]
        super().__init__(
            DiabetesT2DrozdovModel,
            init,
            utils.ensure2D(groundTruthObservation),
            0,
            600,
            1,
            [0, 1, 2, 3, 4, 5],
            idealX,
            learningWindow,
        )
