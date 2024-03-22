from .. import utils
from . import base
from .base import GeneralizedODESystemProblem


class TumorModel:
    def __init__(self, initState, lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE):
        self.initState = initState
        self.lambdap = lambdap
        self.K = K
        self.kqpp = kqpp
        self.kpq = kpq
        self.gammap = gammap
        self.gammaq = gammaq
        self.deltaqp = deltaqp
        self.KDE = KDE
        self.size = 4

    def __call__(self, z, t):
        C, P, Q, QP = z
        Pstar = P + Q + QP
        dCdt = -self.KDE * C
        dPdt = (
            self.lambdap * P * (1 - Pstar / self.K)
            + self.kqpp * QP
            - self.kpq * P
            - self.gammap * C * self.KDE * P
        )
        dQdt = self.kpq * P - self.gammaq * C * self.KDE * Q
        dQpdt = self.gammaq * C * self.KDE * Q - self.kqpp * QP - self.deltaqp * QP
        dzdt = [dCdt, dPdt, dQdt, dQpdt]
        return dzdt

    def withInitState(self, newInitState):
        return TumorModel(newInitState, *self.getModelParams())

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def postprocess(self, z):
        return z[1] + z[2] + z[3]

    def scalarize(self, z):
        return z

    def getModelParams(self):
        return [
            self.lambdap,
            self.K,
            self.kqpp,
            self.kpq,
            self.gammap,
            self.gammaq,
            self.deltaqp,
            self.KDE,
        ]


class TumorModelWD:
    def __init__(self, initState, lambdaP, K, kQ0P, kPH, kPH0, gamma, deltaQH, KDE):
        self.initState = initState
        self.lambdaP = lambdaP
        self.K = K
        self.kQ0P = kQ0P
        self.kPH = kPH
        self.kPH0 = kPH0
        self.gamma = gamma
        self.deltaQH = deltaQH
        self.KDE = KDE
        self.size = 4

    def __call__(self, z, t):
        P0 = self.initState[1]
        C, P, QR, QH = z
        Pstar = P + QR + QH
        dCdt = -self.KDE * C
        dPdt = (
            self.lambdaP * P * (1 - Pstar / self.K)
            + self.kQ0P * QR
            - self.kPH * P
            - self.gamma * C * P
        )
        dQHdt = (
            self.kPH * P
            - self.kPH0 * (P0 - P) * QH
            - self.deltaQH * QH
            - self.gamma * C * QH
        )
        dQRdt = self.kPH0 * (P0 - P) * QH - self.kQ0P * QR - self.gamma * C * QR
        dzdt = [dCdt, dPdt, dQRdt, dQHdt]
        return dzdt

    def withInitState(self, newInitState):
        return TumorModel(newInitState, *self.getModelParams())

    def nudging(self, z, observation):
        scale = observation / self.postprocess(z)
        return z * (scale - 1)

    def postprocess(self, z):
        return z[1] + z[2] + z[3]

    def scalarize(self, z):
        return z

    def getModelParams(self):
        return [
            self.lambdaP,
            self.K,
            self.kQ0P,
            self.kPH,
            self.kPH0,
            self.gamma,
            self.deltaQH,
            self.KDE,
        ]


class TumorProblem(base.ODESystemProblem):
    def __init__(
        self,
        tumorModel,
        modelingWindow,
        initState,
        idealX,
        learningWindow,
        coupledFields,
        groundTruthObservation,
    ):
        #
        # initState = [0, 4.72795425, 48.51476221, 0]
        #
        # baseLambdap = 6.80157379e-01
        # baseK = 1.60140838e+02
        # baseKqpp = 0.00000001e+00
        # baseKpq = 4.17370748e-01
        # baseGammap = 5.74025981e+00
        # baseGammaq = 1.34300000e+00
        # baseDeltaqp = 6.78279483e-01
        # baseKDE = 9.51318080e-02

        # idealX = array(
        #     [
        #         baseLambdap,
        #         baseK,
        #         baseKqpp,
        #         baseKpq,
        #         baseGammap,
        #         baseGammaq,
        #         baseDeltaqp,
        #         baseKDE,
        #     ]
        # )
        tmin, tmax = modelingWindow
        super().__init__(
            tumorModel,
            initState,
            utils.ensure2D(groundTruthObservation),
            tmin,
            tmax,
            1,
            coupledFields,
            idealX,
            learningWindow,
        )


class TumorProblemWithTherapy(GeneralizedODESystemProblem, TumorProblem):
    # # dla jednego przebiegu
    # predictionBeginning = -10
    # predictionEnding = 50
    # therapyApplicationMoments = [0]

    # dla dwoch przebiegow
    # predictionBeginning = -10
    # predictionEnding = 113
    # therapyApplicationMoments = [0,61]

    # dla trzech przebiegow
    # predictionBeginning = -10
    # predictionEnding = 175
    # therapyApplicationMoments = [0,61,123]
    # therapyApplicationMoments = [0,71,93]

    # dla pieciu przebiegow
    # predictionBeginning = -10
    # predictionEnding = 299
    # therapyApplicationMoments = [0,61,123,185,247]

    def __init__(
        self,
        tumorModel,
        modelingWindow,
        initState,
        idealX,
        learningWindow,
        therapyMoments,
        coupledFields,
        groundTruthObservation,
    ):
        super().__init__(
            tumorModel,
            modelingWindow,
            initState,
            idealX,
            learningWindow,
            coupledFields,
            groundTruthObservation,
        )
        self.therapyMoments = therapyMoments

    def modificationPoints(self):
        return [t for t, _ in self.therapyMoments]

    def modifyState(self, t, state):
        idx = [a == t for a, _ in self.therapyMoments].index(True)
        state[0] = self.therapyMoments[idx][1]
        return state
