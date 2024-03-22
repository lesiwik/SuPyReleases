import numpy as np


def runWithModifier(solver, initState, t0, t1, modificationPoints, stateModifier):
    timePoints = [t0] + modificationPoints + [t1]
    output = {}
    for a, b in zip(timePoints, timePoints[1:], strict=False):
        res = solver(initState, a, b)
        for key, val in res.items():
            if key in output:
                prev = output[key]
                output[key] = np.concatenate((prev[:-1], val))
            else:
                output[key] = val[:]
        if b < t1:
            initState = stateModifier(b, res["states"][-1, :])
    return output
