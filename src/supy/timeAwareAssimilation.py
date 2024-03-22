import adao
import numpy
import scipy
from adao import adaoBuilder

if __name__ == "__main__":
    groundTruthObservation = [
        53.24271645564879,
        55.73408614284355,
        57.813057709561186,
        59.62972666793778,
        61.24840184295388,
        63.859216213539206,
        65.7997799043894,
        67.77926410429723,
        69.28402751342766,
        70.81265484922778,
        72.10088901384385,
        71.29892580495805,
        67.95073464005661,
        61.331540140946,
        57.63157970225158,
        50.12167244169479,
        45.40360491222454,
        41.49356685035813,
        37.93253743983854,
        36.1688656152416,
        35.02068670905743,
        33.67645650462805,
        32.581933361277734,
        31.72173449803619,
        30.876810588310185,
        29.843469649983355,
        28.78468810561838,
        27.558211558432504,
        26.884196350458918,
        26.12727674566478,
        25.65886011913788,
        25.285974453184487,
        24.9173266951422,
        24.374992914197623,
        24.147445381985015,
        23.833506139271858,
        23.73825937595852,
        23.643266710167445,
        23.63563916344681,
        23.628013239989844,
        23.619802542997036,
        23.7863419394951,
        23.778094619372666,
        24.03227660799511,
        24.465129968143838,
        24.992959690942634,
        25.437238636013635,
        26.070659685386996,
        26.715782904311283,
        27.370167860875657,
        28.732613710650547,
        29.734825400376987,
        31.597039003311526,
        33.098010640853595,
        35.32361261509472,
        37.41004168571596,
        39.57705260790178,
        41.06724823357617,
        43.37195516176883,
        45.76131544560512,
        48.517302035576286,
    ]

    initState = [0, 4.72795425, 48.51476221, 0]

    baseLambdap = 6.80157379e-01
    baseK = 1.60140838e02
    baseKqpp = 0.00000001e00
    baseKpq = 4.17370748e-01
    baseGammap = 5.74025981e00
    baseGammaq = 1.34300000e00
    baseDeltaqp = 6.78279483e-01
    baseKDE = 9.51318080e-02

    idealX = numpy.array(
        [
            baseLambdap,
            baseK,
            baseKqpp,
            baseKpq,
            baseGammap,
            baseGammaq,
            baseDeltaqp,
            baseKDE,
        ]
    )

    # our tumor growth model
    def rhs(z, t, lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE):
        C, P, Q, QP = z
        Pstar = P + Q + QP

        dCdt = -KDE * C
        dPdt = (
            lambdap * P * (1 - Pstar / K) + kqpp * QP - kpq * P - gammap * C * KDE * P
        )
        dQdt = kpq * P - gammaq * C * KDE * Q
        dQpdt = gammaq * C * KDE * Q - kqpp * QP - deltaqp * QP
        dzdt = [dCdt, dPdt, dQdt, dQpdt]
        return dzdt

    print("Numpy:", numpy.__version__)
    print("Scipy:", scipy.__version__)
    print("Adao:", adao.version)
    numpy.set_printoptions(precision=2, linewidth=5000, threshold=10000)

    Observations = numpy.ravel(groundTruthObservation)
    NbObs = len(Observations)
    TimeBetween2Obs = 1
    TimeWindows = NbObs * TimeBetween2Obs
    NbOfEulerStep = 5  # Number of substeps between 2 obs

    Bounds = [[0.5 * m, 5 * m] for m in idealX]

    def DirectOperator(X):
        lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE = numpy.ravel(X).tolist()
        #
        # State evaluation on the trajectory with z = (C, P, Q, QP)
        # ---------------------------------------------------------
        initState = [0, 4.72795425, 48.51476221, 0]
        dt = 0.1 * TimeBetween2Obs
        # Initial state
        z = numpy.ravel(initState)
        state, time = [], []
        # state.append(z)
        for t in range(NbObs):
            # Substeps between 2 obs
            for step in range(NbOfEulerStep):
                dt = TimeBetween2Obs * step / NbOfEulerStep
                # Use rhs to calculate dz/dt
                dzdt = numpy.ravel(
                    rhs(z, None, lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE)
                )
                # dt/dt = (z(t+1) - z(t)) / dt ===> z(t+1) = z(t) + (dz/dt) * dt
                z = z + numpy.ravel(dzdt) * dt
            state.append(z)
            time.append((t + 1) * TimeBetween2Obs)
        #
        # Relation between the state "z" and the observation
        # --------------------------------------------------
        Pstar = [z[1] + z[2] + z[3] for z in state]
        #
        # Pstar = [z[1]+z[2]/10+z[3] for z in state]
        #
        # Pstar = [6*z[1]+25 for z in state]
        # Pstar = [
        #     3 * (6 * state[i][1] + 25 + 10 * numpy.log(i / 3 + 1)) - 130
        #     for i in range(len(state))
        # ]
        #
        return Pstar

    def PlotOneSerie(__Observations, __Title=""):
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (10, 4)
        #
        plt.figure()
        plt.plot(__Observations, "k-")
        plt.title(__Title, fontweight="bold")
        plt.xlabel("Step")
        plt.ylabel("Value")

    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(DirectOperator(idealX), "Simulation for idealX")

    ### First try: using "*idealX*" as a first guess
    case = adaoBuilder.New("")
    #
    case.setBackground(Vector=idealX)
    case.setBackgroundError(ScalarSparseMatrix=1.0)
    #
    case.setObservationOperator(OneFunction=DirectOperator)
    case.setObservation(Vector=Observations)
    case.setObservationError(ScalarSparseMatrix=0.1**2)
    #
    case.setAlgorithmParameters(
        Algorithm="3DVAR",
        Parameters={
            "Bounds": Bounds,
            "StoreSupplementaryCalculations": [
                "Analysis",
                "CostFunctionJ",
                "CurrentState",
            ],
        },
    )
    case.setObserver(
        Info="  J =",
        Template="ValuePrinter",
        Variable="CostFunctionJ",
    )
    case.execute()
    Xa = case.get("Analysis")
    #
    print("")
    print("Optimal:", Xa[-1])
    print("idealX :", idealX)
    print("")
    #

    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(DirectOperator(idealX), "Simulation for idealX")
    PlotOneSerie(DirectOperator(Xa[-1]), "Simulation for Xa")

    # Second try: using values from Ribba, Table1, PCV column as a first guess
    # Notice: no change of the initial state in the tumor model simulation
    Xtab1pcv = numpy.array(
        [  # Values from Ribba, Table1, PCV
            0.121,  # lambdap
            100.0,  # K
            0.0031,  # kqpp
            0.0295,  # kqp
            0.729,  # gammap=gammaq=gamma
            0.729,  # gammap=gammaq=gamma
            0.00867,  # deltaqp
            0.24,  # KDE
        ]
    )
    #
    case = adaoBuilder.New("")
    #
    case.setBackground(Vector=Xtab1pcv)
    case.setBackgroundError(ScalarSparseMatrix=1.0)
    #
    case.setObservationOperator(OneFunction=DirectOperator)
    case.setObservation(Vector=Observations)
    case.setObservationError(ScalarSparseMatrix=0.1**2)
    #
    case.setAlgorithmParameters(
        Algorithm="3DVAR",
        Parameters={
            # "MaximumNumberOfIterations":25,
            "Bounds": Bounds,
            "StoreSupplementaryCalculations": [
                "Analysis",
                "CostFunctionJ",
                "CurrentState",
            ],
        },
    )
    case.setObserver(
        Info="  J =",
        Template="ValuePrinter",
        Variable="CostFunctionJ",
    )
    case.execute()
    Xa = case.get("Analysis")
    #
    print("")
    print("Analysis:", Xa[-1])
    print("idealX  :", idealX)
    print("")

    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(DirectOperator(Xtab1pcv), "Simulation for Xtab1pcv")
    PlotOneSerie(DirectOperator(Xa[-1]), "Simulation for Xa")

    # Part 2 - Dynamical view : building of simulation and observation operator(s)
    # This view consider that the operators can change at each observation time window.

    Yobs = groundTruthObservation

    TimeBetween2Obs = 1
    NbOfEulerStep = 5  # Number of substeps between 2 obs

    Bounds = [[0.5 * m, 5 * m] for m in idealX]

    def H(Z):
        """Observation operator."""
        # Z <=> [state, parameters]
        #
        __Z = numpy.ravel(Z)
        Pstar = __Z[1] + __Z[2] + __Z[3]
        #
        return numpy.array(Pstar)

    def F(Z):
        """Evolution operator."""
        # Z = [state, parameters]
        #
        __Z = numpy.ravel(Z)
        #
        lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE = __Z[4:]
        #
        dt = 0.1 * TimeBetween2Obs
        # Initial state
        z = numpy.ravel(__Z[0:4])
        state = None
        # state.append(z)
        for step in range(NbOfEulerStep):
            dt = TimeBetween2Obs * step / NbOfEulerStep
            # Use rhs to calculate dz/dt
            dzdt = numpy.ravel(
                rhs(z, None, lambdap, K, kqpp, kpq, gammap, gammaq, deltaqp, KDE)
            )
            # dt/dt = (z(t+1) - z(t)) / dt ===> z(t+1) = z(t) + (dz/dt) * dt
            z = z + numpy.ravel(dzdt) * dt
            state = z
        #
        __Znpu = state.tolist() + __Z[4:].tolist()
        #
        return __Znpu

    case = adaoBuilder.New("")
    #
    case.setObservationOperator(OneFunction=H)
    case.setObservationError(ScalarSparseMatrix=1.0)
    #
    case.setEvolutionModel(OneFunction=F)
    case.setEvolutionError(ScalarSparseMatrix=0.1**2)
    #
    case.setAlgorithmParameters(
        Algorithm="KalmanFilter",
        Parameters={
            "StoreSupplementaryCalculations": [
                "Analysis",
                "APosterioriCovariance",
                "SimulatedObservationAtCurrentAnalysis",
            ],
        },
    )
    #
    # Loop to obtain an analysis at each observation arrival
    #
    XaStep = initState + idealX.tolist()
    VaStep = numpy.identity(len(XaStep))
    for i in range(1, len(Yobs)):
        case.setBackground(Vector=XaStep)
        case.setBackgroundError(Matrix=VaStep)
        case.setObservation(Vector=Yobs[i])
        case.execute(nextStep=True)
        XaStep = case.get("Analysis")[-1]
        VaStep = case.get("APosterioriCovariance")[-1]
    #
    Xa = case.get("Analysis")
    Pa = case.get("APosterioriCovariance")
    Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
    #
    print("")
    # print("  Optimal state and parameters\n",Xa)
    # print("  Final a posteriori variance:",Pa[-1])
    print("Simulated observations:\n", Xs)
    print("")

    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(Xs, "Simulated observations")

    # Changing for non-linear estimation

    case = adaoBuilder.New("")
    #
    case.setObservationOperator(OneFunction=H)
    case.setObservationError(ScalarSparseMatrix=1.0)
    #
    case.setEvolutionModel(OneFunction=F)
    case.setEvolutionError(ScalarSparseMatrix=0.1**2)
    #
    case.setAlgorithmParameters(
        Algorithm="ExtendedKalmanFilter",
        Parameters={
            "StoreSupplementaryCalculations": [
                "Analysis",
                "APosterioriCovariance",
                "SimulatedObservationAtCurrentAnalysis",
            ],
        },
    )
    #
    # Loop to obtain an analysis at each observation arrival
    #
    XaStep = initState + idealX.tolist()
    VaStep = numpy.identity(len(XaStep))
    for i in range(1, len(Yobs)):
        case.setBackground(Vector=XaStep)
        case.setBackgroundError(Matrix=VaStep)
        case.setObservation(Vector=Yobs[i])
        case.execute(nextStep=True)
        XaStep = case.get("Analysis")[-1]
        VaStep = case.get("APosterioriCovariance")[-1]
    #
    Xa = case.get("Analysis")
    Pa = case.get("APosterioriCovariance")
    Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
    #
    print("")
    # print("  Optimal state and parameters\n",Xa)
    # print("  Final a posteriori variance:",Pa[-1])
    print("Simulated observations:\n", Xs)
    print("")

    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(Xs, "Simulated observations")

    # Part 3

    case = adaoBuilder.New("")
    #
    case.setObservationOperator(OneFunction=H)
    case.setObservationError(ScalarSparseMatrix=1.0)
    #
    case.setEvolutionModel(OneFunction=F)
    case.setEvolutionError(ScalarSparseMatrix=0.1**2)
    #
    case.setAlgorithmParameters(
        Algorithm="ExtendedKalmanFilter",
        Parameters={
            "StoreSupplementaryCalculations": [
                "Analysis",
                "APosterioriCovariance",
                "SimulatedObservationAtCurrentAnalysis",
            ],
        },
    )
    #
    # Loop to obtain an analysis at each observation arrival
    #
    XaStep = initState + idealX.tolist()
    VaStep = numpy.identity(len(XaStep))
    for i in range(1, len(Yobs)):
        case.setBackground(Vector=XaStep)
        case.setBackgroundError(Matrix=VaStep)
        case.setObservation(Vector=Yobs[i])
        case.execute(nextStep=True)
        XaStep = case.get("Analysis")[-1]
        VaStep = case.get("APosterioriCovariance")[-1]
        # Modification of the parameters
        # XaStep[5] = XaStep[5]*10
        if i == 10:
            XaStep[0] = 1

    #
    Xa = case.get("Analysis")
    Pa = case.get("APosterioriCovariance")
    Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
    #
    print("")
    # print("  Optimal state and parameters\n",Xa)
    # print("  Final a posteriori variance:",Pa[-1])
    print("Simulated observations:\n", Xs)
    print("")
    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(Xs, "Simulated observations")

    # Part 4 - 4DVAR

    case = adaoBuilder.New("")
    #
    case.setObservationOperator(OneFunction=H)
    case.setObservationError(ScalarSparseMatrix=1.0e-2)
    case.setBackgroundError(ScalarSparseMatrix=1.0)
    #
    case.setEvolutionModel(OneFunction=F)
    case.setEvolutionError(ScalarSparseMatrix=1.0)
    #
    case.setAlgorithmParameters(
        Algorithm="4DVAR",
        Parameters={
            "MaximumNumberOfIterations": 15,
            "StoreSupplementaryCalculations": [
                "Analysis",
                "CurrentState",
            ],
        },
    )
    #
    # Loop to obtain an analysis at each observation arrival
    #
    XaStep = initState + idealX.tolist()
    VaStep = numpy.identity(len(XaStep))
    for i in range(5, 20):  # len(Yobs)):
        case.setBackground(Vector=XaStep)
        case.setObservation(VectorSerie=Yobs[i - 5 : i])
        case.execute(nextStep=True)
        XaStep = F(case.get("Analysis")[-1])
        # Modification of the parameters
        # XaStep[5] = XaStep[5]*10
        # if i==10:
        # XaStep[0] = 1

    #
    Xa = case.get("Analysis")
    # Pa = case.get("APosterioriCovariance")
    # Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
    #
    print("")
    # print("  Optimal state and parameters\n",Xa)
    # print("  Final a posteriori variance:",Pa[-1])
    # print("Simulated observations:\n",Xs)
    print("")
    PlotOneSerie(Observations, "Observations")
    # PlotOneSerie(Xs,"Simulated observations")

    # Part 5 - EnKF

    case = adaoBuilder.New("")
    #
    case.setObservationOperator(OneFunction=H)
    case.setObservationError(ScalarSparseMatrix=1.0e-2)
    #
    case.setEvolutionModel(OneFunction=F)
    case.setEvolutionError(ScalarSparseMatrix=1.0)
    #
    case.setAlgorithmParameters(
        Algorithm="EnsembleKalmanFilter",
        Parameters={
            "StoreSupplementaryCalculations": [
                "Analysis",
                "APosterioriCovariance",
                "SimulatedObservationAtCurrentAnalysis",
            ],
        },
    )
    #
    # Loop to obtain an analysis at each observation arrival
    #
    XaStep = initState + idealX.tolist()
    VaStep = numpy.identity(len(XaStep))
    for i in range(1, len(Yobs)):
        case.setBackground(Vector=XaStep)
        case.setBackgroundError(Matrix=VaStep)
        case.setObservation(Vector=Yobs[i])
        case.execute(nextStep=True)
        XaStep = case.get("Analysis")[-1]
        VaStep = case.get("APosterioriCovariance")[-1]
        # Modification of the parameters
        # XaStep[5] = XaStep[5]*10
        if i == 10:
            XaStep[0] = 1

    #
    Xa = case.get("Analysis")
    Pa = case.get("APosterioriCovariance")
    Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
    #
    print("")
    # print("  Optimal state and parameters\n",Xa)
    # print("  Final a posteriori variance:",Pa[-1])
    print("Simulated observations:\n", Xs)
    print("")
    PlotOneSerie(Observations, "Observations")
    PlotOneSerie(Xs, "Simulated observations")
