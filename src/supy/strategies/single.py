import numpy as np
from scipy.integrate import odeint

from .. import trace, utils


def singleModelSolver(model, t0, t1):
    trace.registerCall()
    n = t1 - t0 + 1
    ts = np.linspace(t0, t1, n)
    # trajectory = solve_ivp(
    #     dt, (t0, t1), np.array(init + initCoeffs), t_eval=ts, method="RK45"
    # )
    # res = solve_ivp(
    #     lambda a, b: model(b, a),
    #     (t0, t1),
    #     np.array(model.initState),
    #     t_eval=ts,
    #     method="RK45",
    # )
    # trajectory = np.transpose(res.y)
    trajectory = odeint(model, model.initState, ts)
    s = np.array([model.postprocess(z) for z in trajectory])
    return {"data": s, "states": trajectory}


class SingleModelRunner:
    def __init__(self, assimilationAlgorithm, name=None):
        self.name = name or "single"
        self.assimilationAlgorithm = assimilationAlgorithm

    def __call__(self, experiment, subModelsParams):
        problem = experiment.problem
        vars = self.assimilationAlgorithm(
            problem.solver,
            problem.referenceModelParams,
            problem.postprocessedGroundTruth,
            problem.postprocess,
            experiment.params,
        )
        return problem.solver(*vars)

    def outputVars(self, experiment):
        T = utils.getTimeStepCount(experiment.problem)
        return [
            ("data", T, experiment.problem.outDim),
            ("states", T, experiment.problem.stateDim),
        ]
