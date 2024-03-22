from supy.assimilation.adao import solve_adao


def test_adao_works(parabola):
    config = {
        "evaluation_budget": 40,
    }

    result = solve_adao(parabola.problem, config)
    parabola.assert_solution_good_enough(result, epsilon=1e-5)


def test_adao_works_with_multiple_objectives(sincos):
    config = {
        "evaluation_budget": 40,
    }

    result = solve_adao(sincos.problem, config)
    sincos.assert_solution_good_enough(result, epsilon=1e-6)
