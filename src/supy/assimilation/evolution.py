import contextlib
import dataclasses
import random
import warnings
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.termination import TerminateIfAny, Termination
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from supy.assimilation.problem import AssimilationProblem, AssimilationResult
from supy.utils import stopwatch, traced, zip_to_dict


def runEvolution(model, avgModelParams, groundTruth, postprocessFunction, evolParams):
    class AssimilationProblem(Problem):
        def __init__(self):
            proc = evolParams["deviation"]
            lowerBounds = [x * (1 - proc) for x in avgModelParams]
            upperBounds = [x * (1 + proc) for x in avgModelParams]
            super().__init__(
                n_var=len(avgModelParams),
                n_obj=1,
                n_constr=0,
                xl=lowerBounds,
                xu=upperBounds,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            res = []
            for i in range(x.shape[0]):
                z = postprocessFunction(model(*x[i, :])["data"])
                res.append(np.linalg.norm(z - np.array(groundTruth)))

            out["F"] = np.column_stack([res])

    problem = AssimilationProblem()
    algorithm = GA(
        pop_size=evolParams["populationSize"],
        eliminate_duplicates=False,
        save_history=True,
    )

    seed = random.randint(1, 100)
    res = minimize(
        problem,
        algorithm,
        termination=(
            "n_gen",
            evolParams["evaluationBudget"] / evolParams["populationSize"],
        ),
        seed=seed,
        verbose=True,
    ).X
    return res


def _termination_criteria(config) -> Termination:
    by_quality = get_termination("soo")
    by_evals = get_termination("n_eval", config["evaluation_budget"])
    by_time = get_termination("time", config["time_budget"])
    return TerminateIfAny(by_quality, by_evals, by_time)


def _build_args(
    alg: str, config: dict[str, Any], pop_size: bool = False
) -> dict[str, Any]:
    known_params = {"evolution.population_size"}
    relevant_params = {"evolution.algorithm", "evolution.extra"}
    args = {}

    if pop_size:
        relevant_params.add("evolution.population_size")
        with contextlib.suppress(KeyError):
            population_size = config["evolution.population_size"]
            args = {"pop_size": population_size}

    for key in config:
        if key.startswith("evolution.") and key not in relevant_params:
            if key in known_params:
                msg = f"Algorithm '{alg}' does not use parameter '{key}'"
            else:
                msg = f"Unknown parameter: '{key}'"
            warnings.warn(msg, stacklevel=4)

    return args


def _choose_algorithm(config) -> Algorithm:
    extra = config.get("evolution.extra", {})

    match alg := config["evolution.algorithm"]:
        case "GA":
            args = _build_args(alg, config, pop_size=True)
            return GA(**args, **extra)

        case "DE":
            from pymoo.algorithms.soo.nonconvex.de import DE

            args = _build_args(alg, config, pop_size=True)
            return DE(**args, **extra)

        case "nelder-mead":
            from pymoo.algorithms.soo.nonconvex.nelder import NelderMead

            args = _build_args(alg, config)
            return NelderMead(**args, **extra)

        case "PSO":
            from pymoo.algorithms.soo.nonconvex.pso import PSO

            args = _build_args(alg, config, pop_size=True)
            return PSO(**args, **extra)

        case "pattern-search":
            from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch

            args = _build_args(alg, config)
            return PatternSearch(**args, **extra)

        case "ES":
            from pymoo.algorithms.soo.nonconvex.es import ES

            args = _build_args(alg, config)
            return ES(**args, **extra)

        case "SRES":
            from pymoo.algorithms.soo.nonconvex.sres import SRES

            args = _build_args(alg, config)
            return SRES(**args, **extra)

        case "ISRES":
            from pymoo.algorithms.soo.nonconvex.isres import ISRES

            args = _build_args(alg, config)
            return ISRES(**args, **extra)

        case "CMAES":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

            # In case of CMAES, pymoo uses an implementation from external library,
            # which initializes numpy's random seeds even if they are not passed
            # explicitly. To suppress this behaviour, seed with value ``np.nan`` needs
            # to be passed: https://github.com/CMA-ES/pycma/issues/221. However, the
            # only way to pass it to CMAES implementation is through keyword argument
            # `seed` of CMAES class in pymoo, which has the side effect of passing it to
            # pymoo itself, which then tries to use this seed value directly, and fails,
            # since ``np.nan`` is not a valid argument to ``np.random.seed``.
            #
            # To bypass this issue, we explicitly pass a seed with value dependent on
            # the global seed.
            seed = np.random.randint(0, 2**32)
            args = _build_args(alg, config)
            return CMAES(**args, seed=seed, **extra)

        case "G3PCX":
            from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX

            args = _build_args(alg, config, pop_size=True)
            return G3PCX(**args, **extra)

        case _:
            raise ValueError(f"Unknown evolutionary algorithm: '{alg}'")


class _PymooProblem(ElementwiseProblem):
    def __init__(self, problem: AssimilationProblem):
        lower, upper = zip(*list(problem.param_bounds.values()), strict=True)
        dim = len(problem.param_bounds)
        super().__init__(n_var=dim, n_obj=1, n_constr=0, xl=lower, xu=upper)

        self.problem = problem

    def _evaluate(self, x, out, *args, **kwarg):
        values = self.problem.model(*x)
        out["F"] = self.distance(values, self.problem.ground_truth)

    def distance(self, x, y):
        # Frobenius norm of the 2D difference array
        output_vars = self.problem.ground_truth.keys()
        diffs = [x[n] - y[n] for n in output_vars]
        return np.linalg.norm(diffs)


def solve_evolution(problem: AssimilationProblem, config) -> AssimilationResult:
    r"""
    Solve an assimilation problem using evolutionary algorithms.

    Parameters
    ----------
    problem : AssimilationProblem
        Instance of a problem to solve.

    config : dict
        Configuration parameters of the evolutionary search. Currently supported
        parameters:

        - 'evolution.algorithm': evolutionary algorithm used for optimization:

          - 'GA': Basic :math:`\mu + \lambda` genetic algorithm
          - 'DE': Differential Evolution
          - 'nelder-mead': Nelder-Mead function minimization algorithm
          - 'PSO': Particle Swarm optimization
          - 'pattern-search': Hooke and Jeeves Pattern Search
          - 'ES': Evolutionary Strategy
          - 'SRES': Stochastic Ranking Evolutionary Strategy
          - 'ISRES': Improved Stochastic Ranking Evolutionary Strategy
          - 'CMAES': Covariance Matrix Adaptation Evolution Strategy
          - 'G3PCX': Implementation of Parent-centric crossover (PCX) operator using G3
            model

        - 'evolution.population_size' (optional): size of the population, used by many,
          but not all the evolutionary algorithms. If not specified, a default value,
          depending on the algorithm, will be used.

          ================ =====
            Algorithm      used?
          ================ =====
          'GA'              yes
          'DE'              yes
          'nelder-mead'     no
          'PSO'             yes
          'pattern-search'  no
          'ES'              no
          'SRES'            no
          'ISRES'           no
          'CMAES'           no
          'G3PCX'           yes
          ================ =====

        - 'evaluation_budget': maximum number of objective function evaluation
          allowed during the assimilation process
        - 'time_budget': maximum allowed walltime (in seconds) the assimilation
          process is allowed to take

    Returns
    -------
    AssimilationResult
        Optimal parameter values and runtime statistics.

    Notes
    -----
    This function uses implementations of multiple single-objective evolutionary
    algorithms from the pymoo library [1]_.

    References
    ----------
    .. [1] J. Blank and K. Deb, "pymoo: Multi-Objective Optimization in Python",
       IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567
    """
    with stopwatch() as total:
        model = traced(problem.model)
        traced_problem = dataclasses.replace(problem, model=model)
        instance = _PymooProblem(traced_problem)
        algorithm = _choose_algorithm(config)
        termination = _termination_criteria(config)

        result = minimize(instance, algorithm, termination)
        param_values = zip_to_dict(problem.params, result.X)

    return AssimilationResult(
        param_values,
        model.call_count,
        total_time=total.elapsed_time,
        model_time=model.elapsed_time,
    )
