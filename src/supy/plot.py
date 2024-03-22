import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class PlotItem:
    def __init__(self, key, label, style, color, idx=None):
        self.key = key
        self.label = label
        self.style = style
        self.color = color
        self.idx = idx


class PlotDescription:
    def __init__(
        self, items, title, axesLabels, legendAttr, xAxis, vLines, plotFunc=None
    ):
        self.items = items
        self.title = title
        self.legendAttr = legendAttr
        self.axesLabels = axesLabels
        self.xAxis = xAxis
        self.vLines = vLines
        self.plotFunc = plotFunc


@dataclass
class DataWithError:
    data: Any
    error: Any


def plotFigure(descriptions, data, nCols, size):
    f = plt.figure(figsize=(size[0], size[1]))
    nRows = math.ceil(len(descriptions) / nCols)

    def getSeries(key):
        return key(data) if callable(key) else data[key]

    for i, plot in enumerate(descriptions):
        if plot.plotFunc is not None:
            plot.plotFunc(f, data, plot, (nRows, nCols, i))
        else:
            ax = f.add_subplot(nRows, nCols, i + 1)
            for it in plot.items:
                arg = dict(color=it.color, label=it.label)
                xAxis = data[plot.xAxis] if isinstance(plot.xAxis, str) else plot.xAxis
                if isinstance(it.key, DataWithError):
                    vals = getSeries(it.key.data)
                    err = getSeries(it.key.error)
                    if it.idx is not None:
                        vals = vals[:, it.idx]
                        err = err[:, it.idx]
                    arg["yerr"] = np.array(err).flatten()
                    ax.errorbar(xAxis, vals, fmt=it.style, **arg)
                else:
                    vals = getSeries(it.key)
                    if it.idx is not None:
                        vals = vals[:, it.idx]
                    ax.plot(xAxis, vals, it.style, **arg)
            ax.set_xlabel(plot.axesLabels[0])
            ax.set_ylabel(plot.axesLabels[1])
            ax.set_title(plot.title, fontdict={"fontsize": 12, "fontweight": "bold"})
            for li in plot.vLines:
                ax.axvline(li)
            ax.legend(**plot.legendAttr)
    f.tight_layout()
    plt.show()
