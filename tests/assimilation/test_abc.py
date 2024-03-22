import pytest

from supy.assimilation.abc import solve_abc


@pytest.mark.slow()
def test_abc_works(parabola):
    config = {
        "evaluation_budget": 200,
        "time_budget": 1,
        "abc.population_size": 10,
        "abc.minimum_epsilon": 0.1,
        "abc.sample_size": 100,
    }

    result = solve_abc(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.015)


@pytest.mark.slow()
def test_abc_works_with_multiple_objectives(sincos):
    config = {
        "evaluation_budget": 200,
        "time_budget": 1,
        "abc.population_size": 10,
        "abc.minimum_epsilon": 0.1,
        "abc.sample_size": 100,
    }

    result = solve_abc(sincos.problem, config)
    sincos.assert_solution_good_enough(result, epsilon=0.015)
