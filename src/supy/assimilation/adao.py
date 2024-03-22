import random

import numpy as np
from adao import adaoBuilder
from numpy.typing import NDArray

from supy.assimilation.problem import AssimilationProblem, AssimilationResult
from supy.interval import Interval
from supy.utils import stopwatch, traced, zip_to_dict

from ..strategies.single import singleModelSolver


def runTimeAwareADAO(model, t0, t1, avgModelParams, groundTruth, adaoParams):
    ADAO = adaoBuilder.New("")
    ADAO.setObservationError(ScalarSparseMatrix=0.1**2)
    ADAO.setObservationOperator(OneFunction=model.postprocess)
    ADAO.setAlgorithmParameters(
        Algorithm=adaoParams["adaoAlgorithm"],  # algorithm passed as a adao parameter
        Parameters={
            "StoreSupplementaryCalculations": [
                "Analysis",
                "APosterioriCovariance",
            ],
            "MaximumNumberOfIterations": adaoParams["evaluationBudget"],
            "MaximumNumberOfFunctionEvaluations": adaoParams["evaluationBudget"],
            **adaoParams.get("extra", {}),
        },
    )

    def evolutionOperator(t):
        def step(z):
            z = np.array(z).flatten()
            newModel = model.withInitState(z)
            return singleModelSolver(newModel, t, t + 1)["states"][-1, :]

        return step

    XaStep = model.initState
    VaStep = np.diag([0.1 for _ in XaStep])
    for i in range(1, t1 - t0 + 1):
        ADAO.setBackground(Vector=XaStep)
        ADAO.setBackgroundError(Matrix=VaStep)
        ADAO.setObservation(Vector=groundTruth[i])

        ADAO.setEvolutionModel(OneFunction=evolutionOperator(t0 + i))
        ADAO.setEvolutionError(ScalarSparseMatrix=1e-5)

        ADAO.execute(nextStep=True)
        XaStep = ADAO.get("Analysis")[-1]
        VaStep = ADAO.get("APosterioriCovariance")[-1]
    res = np.array([xa[0] for xa in ADAO.get("Analysis")])
    return res


def runADAO(model, avgModelParams, groundTruth, postprocessFunction, adaoParams):
    def observationOperator(args):
        args = np.asarray(args).reshape(-1)
        return postprocessFunction(model(*args)["data"])

    proc = adaoParams["deviation"]
    bounds = [[x * (1 - proc), x * (1 + proc)] for x in avgModelParams]
    boxBounds = [[-1e10, 1e10] for x in avgModelParams]
    Xb = [random.uniform(*b) for b in bounds]
    Yobs = np.array(groundTruth)
    XbError = [1e6 * number**2 for number in Xb]
    YobsError = [number**2 for number in Yobs]
    ADAO = adaoBuilder.New("")
    ADAO.setBackground(Vector=Xb, Stored=True)
    ADAO.setBackgroundError(DiagonalSparseMatrix=XbError)
    ADAO.setObservation(Vector=Yobs, Stored=True)
    ADAO.setObservationError(DiagonalSparseMatrix=YobsError)
    ADAO.setObservationOperator(OneFunction=observationOperator)
    ADAO.setAlgorithmParameters(
        Algorithm=adaoParams["adaoAlgorithm"],
        Parameters={
            "StoreSupplementaryCalculations": [
                "CurrentState",
            ],
            "MaximumNumberOfIterations": adaoParams["evaluationBudget"],
            "MaximumNumberOfFunctionEvaluations": adaoParams["evaluationBudget"],
            "Bounds": bounds,
            "BoxBounds": boxBounds,
            **adaoParams.get("extra", {}),
        },
    )
    if adaoParams.get("adao.printState", True):
        ADAO.setObserver(
            Info="  Intermediate state at the current iteration:",
            Template="ValuePrinter",
            Variable="CurrentState",
        )
    ADAO.execute()
    res = ADAO.get("Analysis")[-1]
    return res


def _max_modulus(bounds: Interval) -> float:
    a, b = bounds
    return max(abs(a), abs(b))


def _background_error(param_bounds: list[Interval]):
    magnitudes = [_max_modulus(b) for b in param_bounds]
    return np.array(magnitudes) ** 2


def _observation_error(observation: dict[str, NDArray]):
    # produce numpy array with the same shape as the one returned by _as_single_vector
    # (concatenated observation series), but in each series the values are replaced
    # with the largest absolute value encountered in it.
    magnitudes = [np.abs(data).max() for data in observation.values()]
    observation_length = next(iter(observation.values())).size
    arr = np.repeat(magnitudes, observation_length)

    return arr**2


def _configure_algorithm(case, param_bounds, config):
    # It looks like the default underlying optimization routime (L-BFGS-B from
    # scipy.optimize) uses 1 + dim(param_space) evaluations per iteration, since ADAO
    # uses additional evaluations to compute the gradient, needed by L-BFGS-B.
    param_space_dim = len(param_bounds)
    max_iters = config["evaluation_budget"] // (1 + param_space_dim)

    case.setAlgorithmParameters(
        Algorithm="3DVAR",
        Parameters={
            "MaximumNumberOfIterations": max_iters,
            "Bounds": param_bounds,
            "StoreSupplementaryCalculations": ["CurrentState"],
        },
    )
    if config.get("adao.log_state"):
        case.setObserver(
            Info="  Intermediate state at the current iteration:",
            Template="ValuePrinter",
            Variable="CurrentState",
        )


def _as_single_vector(output: dict[str, NDArray]) -> list[NDArray]:
    vectors = list(output.values())
    return np.concatenate(vectors)


def solve_adao(problem: AssimilationProblem, config) -> AssimilationResult:
    r"""
    Solve an assimilation problem using 3DVAR algorithm from ADAO.

    Parameters
    ----------
    problem : AssimilationProblem
        Instance of a problem to solve.

    config : dict
        Configuration parameters of the 3DVAR algorithm. Currently supported parameters:

        - 'evaluation_budget': maximum number of objective function evaluation allowed
          during the assimilation process

    Returns
    -------
    AssimilationResult
        Optimal parameter values and runtime statistics.

    Notes
    -----
    The 3DVAR algorithm requires background error and observation error covariance
    matrices. In practice, these can be used to appropriately scale the contributions of
    background and observation data in the objective function 3DVAR strives to minimize.
    In our implementation, we attempt to equalize the impact of all the output variables
    by dividing error of each time series in the output variables by the maximum
    absolute value in this series' ground truth. This effectively replaces absolute
    errors by the relative ones, scaled by :math:`l^\infty` norm.

    For observation data::

        [ output_1, output_2, ..., output_N ]

    the observation error is calculated as the point-wise square of::

        [ max_1 .... max_1, max_2 .... max_2, ..., max_N .... max_N]
              k times           k times                k times

    where ``max_i = max(abs(output_i))``, and `k` is the length of the time series
    comprising observation data. The background error is computed similarly, using the
    largest absolute value inside the interval specified for the parameter as the scale.

    Since we do not have a reliable background data (it is chosen randomly from the
    parameters' domain), the background error is further scaled up to reduce its impact
    on the solution.

    There seems to be no way impose a time limit in the ADAO interface, so only the
    evaluation budget is available.
    """
    with stopwatch() as total:

        @traced
        def model(coeffs):
            output = problem.model(*coeffs)
            return _as_single_vector(output)

        param_bounds = list(problem.param_bounds.values())
        init_params = [random.uniform(*b) for b in param_bounds]

        background_error = 1e6 * _background_error(param_bounds)
        observation_error = _observation_error(problem.ground_truth)

        case = adaoBuilder.New()

        case.setBackground(Vector=init_params)
        case.setObservation(Vector=_as_single_vector(problem.ground_truth))
        case.setBackgroundError(DiagonalSparseMatrix=background_error)
        case.setObservationError(DiagonalSparseMatrix=observation_error)
        case.setObservationOperator(OneFunction=model)

        _configure_algorithm(case, param_bounds, config)

        case.execute()
        result = case.get("Analysis")[-1]
        param_values = zip_to_dict(problem.params, result)

    return AssimilationResult(
        param_values,
        model.call_count,
        total_time=total.elapsed_time,
        model_time=model.elapsed_time,
    )
