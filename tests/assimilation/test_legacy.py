from datetime import timedelta

import pytest

from supy.assimilation import runABC, runADAO, runEvolution
from supy.assimilation.problem import AssimilationResult


def _wrap_result(data, case):
    names = case.reference_params.keys()
    values = dict(zip(names, data, strict=True))
    return AssimilationResult(
        values,
        model_calls=0,
        total_time=timedelta(),
        model_time=timedelta(),
    )


@pytest.mark.slow()
def test_legacy_abc_works(parabola):
    abc_params = {
        "deviation": 0.2,
        "computationTime": 1,
        "minimumEpsilon": 0.1,
        "populationSize": 10,
        "evaluationBudget": 200,
        "sampleSize": 100,
    }
    reference_params = list(parabola.reference_params.values())
    ground_truth = parabola.problem.ground_truth["data"]
    res = runABC(
        parabola.model, reference_params, ground_truth, lambda x: x, abc_params
    )

    result = _wrap_result(res, parabola)
    parabola.assert_solution_good_enough(result, 0.015)


def test_legacy_adao_works(parabola):
    abc_params = {
        "deviation": 0.2,
        "adaoAlgorithm": "3DVAR",
        "evaluationBudget": 40,
    }
    reference_params = list(parabola.reference_params.values())
    ground_truth = parabola.problem.ground_truth["data"]
    res = runADAO(
        parabola.model, reference_params, ground_truth, lambda x: x, abc_params
    )

    result = _wrap_result(res, parabola)
    parabola.assert_solution_good_enough(result, 1e-5)


def test_legacy_evolution_works(parabola):
    evol_params = {
        "deviation": 0.2,
        "populationSize": 10,
        "evaluationBudget": 400,
    }
    reference_params = list(parabola.reference_params.values())
    ground_truth = parabola.problem.ground_truth["data"]
    res = runEvolution(
        parabola.model, reference_params, ground_truth, lambda x: x, evol_params
    )

    result = _wrap_result(res, parabola)
    parabola.assert_solution_good_enough(result, 0.015)
