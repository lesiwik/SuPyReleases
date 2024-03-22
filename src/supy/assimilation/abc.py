from collections.abc import Iterable
from datetime import timedelta

import numpy as np
import pyabc

from supy.assimilation.problem import AssimilationProblem, AssimilationResult
from supy.interval import Interval
from supy.utils import stopwatch, traced

sampler = pyabc.sampler.SingleCoreSampler(check_max_eval=True)


def runABC(model, avgModelParams, groundTruth, postprocessFunction, abcParams):
    def ParamRV(baseValue, deviation):
        return pyabc.RV(
            "uniform", baseValue - deviation * baseValue, 2 * baseValue * deviation
        )

    paramNumber = len(avgModelParams)
    print(abcParams)
    prior = pyabc.Distribution(
        **{
            f"param{i}": ParamRV(avgModelParams[i], abcParams["deviation"])
            for i in range(paramNumber)
        }
    )

    def abcModel(params):
        paramList = [params[f"param{i}"] for i in range(paramNumber)]
        return model(*paramList)

    def errorFunction(x, y):
        xx = postprocessFunction(x["data"])
        yy = np.array(y["data"])
        return np.linalg.norm(xx - yy)

    abc = pyabc.ABCSMC(
        abcModel,
        prior,
        errorFunction,
        population_size=abcParams["populationSize"],
        sampler=sampler,
    )
    dbPath = "sqlite://"
    abc.new(dbPath, {"data": groundTruth})
    td = timedelta(seconds=abcParams["computationTime"])
    history = abc.run(
        minimum_epsilon=abcParams["minimumEpsilon"],
        max_walltime=td,
        max_total_nr_simulations=abcParams["evaluationBudget"],
    )
    posterior = pyabc.MultivariateNormalTransition()
    posterior.fit(*history.get_distribution(m=0))

    sample_size = abcParams.get("sampleSize", 1)
    variables = posterior.rvs(sample_size)
    return [variables[f"param{i}"].mean() for i in range(paramNumber)]


def _param_rv(bounds: Interval) -> pyabc.RV:
    return pyabc.RV("uniform", bounds.start, bounds.length)


def _distance_function(variables: Iterable[str]):
    def distance(x, y):
        # Frobenius norm of the 2D difference array
        diffs = [x[n] - y[n] for n in variables]
        return np.linalg.norm(diffs)

    return distance


def solve_abc(problem: AssimilationProblem, config) -> AssimilationResult:
    """
    Solve an assimilation problem using ABC.

    This function uses an implementation of ABC based on Sequential Monte Carlo (SMC)
    from pyABC library.

    Parameters
    ----------
    problem : AssimilationProblem
        Instance of a problem to solve.

    config : dict
        Configuration parameters of the ABC algorithm. Currently supported parameters:

        - 'abc.population_size': size of the population
        - 'abc.minimum_epsilon': threshold value for
        - 'evaluation_budget': maximum number of objective function evaluation allowed
          during the assimilation process
        - 'time_budget': maximum allowed walltime (in seconds) the assimilation
          process is allowed to take

    Returns
    -------
    AssimilationResult
        Optimal parameter values and runtime statistics.

    Notes
    -----
    The time and evaluation limits imposed by the configuration parameters are likely to
    be slightly exceeded.
    """
    with stopwatch() as total:
        variables = {var: _param_rv(bd) for var, bd in problem.param_bounds.items()}
        prior = pyabc.Distribution(**variables)

        output_vars = problem.ground_truth.keys()
        dist_fun = _distance_function(output_vars)

        @traced
        def model(params):
            return problem.model(**params)

        sampler = pyabc.sampler.SingleCoreSampler(check_max_eval=True)
        abc = pyabc.ABCSMC(
            model,
            prior,
            dist_fun,
            sampler=sampler,
            population_size=config["abc.population_size"],
        )
        abc.new("sqlite://", problem.ground_truth)

        history = abc.run(
            minimum_epsilon=config["abc.minimum_epsilon"],
            max_total_nr_simulations=config["evaluation_budget"],
            max_walltime=timedelta(seconds=config["time_budget"]),
        )

        posterior = pyabc.MultivariateNormalTransition()
        posterior.fit(*history.get_distribution())

        sample = posterior.rvs(config["abc.sample_size"])
        param_values = {var: sample[var].mean() for var in problem.params}

    return AssimilationResult(
        param_values,
        model.call_count,
        total_time=total.elapsed_time,
        model_time=model.elapsed_time,
    )
