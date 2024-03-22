import pytest

from supy.assimilation.evolution import solve_evolution


def test_evolution_GA(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "GA",
        "evolution.population_size": 10,
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.015)


def test_evolution_DE(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "DE",
        "evolution.population_size": 30,
        "evolution.extra": {
            "variant": "DE/rand/1/bin",
            "F": 0.8,
            "CR": 0.2,
        },
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.015)


def test_evolution_nelder_mead(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "nelder-mead",
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=1e-8)


def test_evolution_PSO(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "PSO",
        "evolution.population_size": 10,
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.0005)


def test_evolution_pattern_search(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "pattern-search",
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=1e-7)


def test_evolution_ES(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "ES",
        "evolution.extra": {
            "n_offsprings": 40,
        },
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.015)


def test_evolution_SRES(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "SRES",
        "evolution.extra": {
            "n_offsprings": 40,
        },
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.015)


def test_evolution_ISRES(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "ISRES",
        "evolution.extra": {
            "n_offsprings": 20,
        },
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=0.03)


def test_evolution_CMAES(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "CMAES",
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=1e-5)


def test_evolution_G3PCX(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "G3PCX",
        "evolution.population_size": 80,
    }

    result = solve_evolution(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=1e-4)


def test_evolution_works_with_multiple_objectives(sincos):
    config = {
        "evaluation_budget": 400,
        "time_budget": 10,
        "evolution.algorithm": "GA",
        "evolution.population_size": 10,
    }

    result = solve_evolution(sincos.problem, config)
    sincos.assert_solution_good_enough(result, epsilon=0.0005)


def test_evolution_warns_about_unknown_args(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "GA",
        "evolution.population_size": 10,
        "evolution.foo": 666,
    }
    with pytest.warns(UserWarning, match="(?i:unknown).*'evolution.foo'"):
        solve_evolution(parabola.problem, config)


def test_evolution_warns_about_unused_args(parabola):
    config = {
        "evaluation_budget": 400,
        "time_budget": 1,
        "evolution.algorithm": "nelder-mead",
        "evolution.population_size": 666,
    }

    pattern = "'nelder-mead'.*does not use.*'evolution.population_size'"
    with pytest.warns(UserWarning, match=pattern):
        solve_evolution(parabola.problem, config)
