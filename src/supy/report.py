import itertools as it

from .plot import DataWithError, PlotDescription, PlotItem


def withParams(f):
    def wrapper(*args):
        return lambda params, **kwargs: f(params, *args, **kwargs)

    return wrapper


def getter(name, attribute):
    return lambda data: getattr(data[name], attribute)


@withParams
def generalPlot(
    params,
    methods,
    idx,
    title,
    axesLabels,
    legendAttribute,
    vLines,
    withSubmodels=False,
    colors=None,
):
    colors = colors or it.repeat(None)
    items = [
        PlotItem(m.name + ".data", m.name, "-", c, idx)
        for m, c in zip(methods, colors, strict=False)
    ]

    items.append(PlotItem("gt", "Ground Truth", "-", None, idx))
    if withSubmodels:
        number = params["numberOfSubmodels"]
        items += [
            PlotItem(f"submodel{n}.data", f"submodel{n}", "-", None, idx)
            for n in range(number)
        ]

    plotDesc = PlotDescription(items, title, axesLabels, legendAttribute, "t", vLines)
    return [plotDesc]


@withParams
def statePlot(
    params, methods, quantities, axesLabels, legendAttribute, vLines, colors=None
):
    colors = colors or it.repeat(None)
    plots = []
    for m, title in methods:
        items = [
            PlotItem(m.name + ".states", name, "-", c, idx)
            for (name, idx), c in zip(quantities, colors, strict=False)
        ]
        plotDesc = PlotDescription(
            items, title, axesLabels, legendAttribute, "t", vLines
        )
        plots.append(plotDesc)
    return plots


@withParams
def couplingsPlot(params, legendAttribute, vLines, colors=None):
    colors = colors or it.repeat(None)
    indices = range(params["numberOfSubmodels"])
    names = [f"C{i}{j}" for i in indices for j in indices if i != j]
    items = [
        PlotItem("nudged." + n, n, "-", c, 0)
        for n, c in zip(names, colors, strict=False)
    ]
    plotDesc = PlotDescription(
        items, "Couplings", ("time", None), legendAttribute, "t", vLines
    )
    return [plotDesc]


@withParams
def cptWeightsPlot(params, legendAttribute, vLines, colors=None):
    colors = colors or it.repeat(None)
    indices = range(params["numberOfSubmodels"])
    items = [
        PlotItem("cpt.weights", f"M{i + 1}", "-", c, i)
        for i, c in zip(indices, colors, strict=False)
    ]
    r = list(range(params["cpt.iters"] + 1))
    plotDesc = PlotDescription(
        items, "CPT Weights", ("iters", None), legendAttribute, r, vLines
    )
    return [plotDesc]


@withParams
def deviationPlot(
    params, methods, idx, title, axesLabels, legendAttribute, vLines, colors=None
):
    def diff(key):
        return lambda data: data[key] - data["gt"]

    colors = colors or it.repeat(None)
    items = [
        PlotItem(diff(m.name + ".data"), m.name, "-", c, idx)
        for m, c in zip(methods, colors, strict=False)
    ]
    plotDesc = PlotDescription(items, title, axesLabels, legendAttribute, "t", vLines)
    return [plotDesc]


@withParams
def averagePlot(
    params, methods, idx, title, axesLabels, legendAttribute, vLines, colors=None
):
    def avg(name):
        return getter(name, "avg")

    colors = colors or it.repeat(None)
    items = [
        PlotItem(avg(m.name + ".data"), m.name, "-", c, idx)
        for m, c in zip(methods, colors, strict=False)
    ]
    items.append(PlotItem("gt", "Ground Truth", "-", None, idx))

    plotDesc = PlotDescription(items, title, axesLabels, legendAttribute, "t", vLines)
    return [plotDesc]


@withParams
def averageErrorPlot(
    params, methods, idx, title, axesLabels, legendAttribute, vLines, colors=None
):
    def error(name):
        data = getter(name, "avgError")
        error = getter(name, "avgErrorDeviation")
        return DataWithError(data, error)

    colors = colors or it.repeat(None)
    items = [
        PlotItem(error(m.name + ".data"), m.name, "-", c, idx)
        for m, c in zip(methods, colors, strict=False)
    ]

    plotDesc = PlotDescription(items, title, axesLabels, legendAttribute, "t", vLines)
    return [plotDesc]


@withParams
def suiteErrorPlot(params, methods, title, axesLabels, legendAttribute, colors=None):
    def withDeviation(name):
        def errorValue(data):
            return data["single.mean"] - data[name + ".mean"]

        def errorStd(data):
            return (data["single.std"] + data[name + ".std"]) / 2

        return DataWithError(data=errorValue, error=errorStd)

    colors = colors or it.repeat(None)
    items = [
        PlotItem(withDeviation(m.name), m.name, "-", c)
        for m, c in zip(methods, colors, strict=False)
    ]

    plotDesc = PlotDescription(
        items, title, axesLabels, legendAttribute, list(range(len(params))), ()
    )
    return [plotDesc]


@withParams
def suiteErrorDeviationPlot(
    params, methods, title, axesLabels, legendAttribute, colors=None
):
    def withDeviation(name):
        return DataWithError(name + ".mean", name + ".std")

    colors = colors or it.repeat(None)
    items = [
        PlotItem(withDeviation(m.name), m.name, "-", c)
        for m, c in zip(methods, colors, strict=False)
        if m.name != "single"
    ]

    plotDesc = PlotDescription(
        items, title, axesLabels, legendAttribute, list(range(len(params))), ()
    )
    return [plotDesc]


def combine(*plots, **kwargs):
    return lambda params: sum((p(params, **kwargs) for p in plots), [])
