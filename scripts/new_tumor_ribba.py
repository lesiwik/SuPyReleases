# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ribba tumor model with new tools
#
# ---
#
# The purpose of this notebook is to demonstrate new features of `SuPy`,especially
# in the area of defining and executing experiments, as well as reporting their
# results. The computational core - models and optimizers - remains the same (for
# now). Since this core is tightly coupled to the old experiment infrastructure
# and rather inflexible, certain things are a bit awkward to do at this point.
# Some other things are not quite fleshed out yet, necessitating hacking it together
# inside the notebook, thus getting one's hands dirty with the code not covered by
# a tidy veneer of a sane API. The "final product" should be more user-friendly :)
#
# First, a bunch of imports.

# %% jupyter={"source_hidden": true}
import io
import multiprocessing
from contextlib import redirect_stdout

import dill as pickle
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import supy.utils as utils
from supy.amb import nondeterministic, once
from supy.assimilation import runABC, runADAO, runEvolution
from supy.datatree.amb import execution_tree
from supy.experiment import Experiment
from supy.problems.tumor import TumorModel, TumorProblemWithTherapy
from supy.strategies import (
    AssimilatedSuperModelRunner,
    CPTSuperModelRunner,
    SingleModelRunner,
    WeightedSuperModelRunner,
)
from supy.utils import dataFile, set_seed, spawn_task

# %% [markdown]
# Ground truth observation needs to be transformed from radius to volume,
# and scaled appropriately to match the values produced by the model.


# %%
def standardGroundTruth():
    numberOfCycles = 1
    data = np.loadtxt(dataFile("tumor/standard.csv"))
    groundTruthObservation = np.tile(data, numberOfCycles)

    def transform(x):
        scalingFactor = 1500
        return (x**3) * np.pi / 6 / scalingFactor

    return list(map(transform, groundTruthObservation))


# %% [markdown]
# ## Experiment definition
#
#
# #### Introduction
# The "experiment" here stands for a set of computations we want to execute
# and compare their results. They are often similar in structure, with differences
# restricted to some coefficient values, employed optimization method  or other
# configuration, though sometimes these differences can be more significant.
#
# One fairly straightforward model of such experiments is based on parametrizing
# the experiment function. We choose a set of parameters and the sets over which
# they vary, and then execute the cartesian product of all the combinations. One
# downside of such an approach is its relative rigidity - the "shape" of the
# experiment is static (decided a priori by the parametrization), restricted
# to the cartesian product of parameter spaces, or some subset of it, and defined
# globally, in the sense that modifying the set of computations requires changes
# at the highest level.
#
# Experiments in SuPy are modelled as a "non-deterministic functions", inspired
# by McCarthy's [`amb` operator][1], and Haskell [list monads][2].
# A non-deterministic function in SuPy is a function accepting a single argument
# (let's call it `amb`). In its body, it can invoke `amb` with a list of values:
# ```python
# param_val = amb("some name", [1, 2, 3]) # first argument is not important
# ```
# As a result, `amb` will return one of these values. This construction is used
# to express a **choice**, or a **branching point** in the computation.
# A non-deterministic function can have any number of such branching points.
# Future choices, and the computation as a whole, can depend on previously made
# choices. As a result, non-deterministic function dynamically defines a tree of
# possible execution paths. This allows us to overcome the aforementioned
# limitations of the parametrization model:
#
# - Execution paths are defined by the experiment function as it is executed, so
#   they are fully dynamic and can depend on the partial results produced during
#   the execution.
# - We can produce an execution tree of arbitrary shape, not restricted to a
#   cartesian product or any other shape.
# - To introduce another branching point, just add a call to `amb` (local change),
#   no global change needed.
#
# For more information, see `supy.amb` documentation, and usage examples in unit
# tests (`tests/test_amb.py`).
#
# #### Ribba model experiment
#
# What follows is an example experiment definition as a non-deterministic function.
# The overall plan is to try to fit the parameters of the Ribba model to standard
# tumor size data:
#
# - using ADAO or ABC to pre-train the submodels
# - in case of ABC, using `epsilon` value of 0.1 or 0.4
# - perform 2 iterations of each configuration
# - in each iteration, pre-train the submodels once and share them between
#   execution paths "below"
# - train supermodels using ADAO or evolutionary algorithm
# - as supermodels, use `SingleModelRunner` (no supermodel),
#   `AssimilatedSuperModelRunner`, `WeightedSuperModelRunner`
#   or `CPTSuperModelRunner`
#
# The resulting execution tree, with with a total of 48 paths looks like this:
#
# <center>
# <img src="/files/docs/source/_static/img/tree.svg" width="800">
# </center>
#
# Nodes marked `save-pretraining` represent the `"submodel-params"` pseudo-choice,
# used to avoid repeated training of models used as components of supermodels.
# All the other nodes represent a genuine branch in the execution.
#
# [1]: http://community.schemewiki.org/?amb
# [2]: https://wiki.haskell.org/All_About_Monads#The_List_monad


# %%
modelingWindow = (-10, 50)
learningWindow = (10, 20)
therapyMoments = [(0, 1)]
full_gt = standardGroundTruth()


def run_stuff(amb):
    supermodel_types = [
        SingleModelRunner,
        AssimilatedSuperModelRunner,
        WeightedSuperModelRunner,
        CPTSuperModelRunner,
    ]

    # C, P, Q, QP
    coupledFields = [0, 1]
    initState = [0, 4.72795425, 48.51476221, 0]

    referenceSolution = {
        "Lambdap": 6.80157379e-01,
        "K": 1.60140838e02,
        "Kqpp": 0.00000001e00,
        "Kpq": 4.17370748e-01,
        "Gammap": 5.74025981e00,
        "Gammaq": 1.34300000e00,
        "Deltaqp": 6.78279483e-01,
        "KDE": 9.51318080e-02,
    }

    idealX = list(referenceSolution.values())

    genericParams = {
        "populationSize": 20,
        "computationTime": 1000,
        "evaluationBudget": 40,
        "pretrainingBudgetRate": 0.2,
        "numberOfSubmodels": 3,
        "deviation": 0.4,
        # These parameters could also be defined only in the
        # execution paths that need them.
        # nudged only
        "nudged.CMagic": 0.1,
        "nudged.K": 0.1,
        # adao only
        "adaoAlgorithm": "3DVAR",
        # cpt only
        "cpt.iters": 10,
        "cpt.timePoints": [-10, 0, 10, 20, 30, 40, 50],
        # try a couple of different pretraining algorithms
        "pretrainingAlgorithm": amb("pretrain-alg", [runADAO, runABC]),
    }

    algParams = {}
    if genericParams["pretrainingAlgorithm"] == runABC:
        # we might want to try several ABC configurations
        algParams["minimumEpsilon"] = amb("abc.epsilon", [0.4, 0.1])

    tumorParams = dict(**genericParams, **algParams)

    problem = TumorProblemWithTherapy(
        TumorModel,
        modelingWindow,
        initState,
        idealX,
        learningWindow,
        therapyMoments,
        coupledFields,
        full_gt,
    )

    gt = problem.postprocess(full_gt)
    numberOfSubmodels = tumorParams["numberOfSubmodels"]
    scale = tumorParams["pretrainingBudgetRate"]
    pretrainingParams = utils.scaleBudget(tumorParams, scale)
    assimilationAlg = tumorParams["pretrainingAlgorithm"]
    newParams = utils.scaleBudget(tumorParams, 1 - scale)

    # trick: repeat each experiment a couple times by choosing an iteration index
    repTime = 2
    _ = amb("iteration", range(repTime))

    solver = problem.solver
    postprocess = problem.postprocess

    def computeSubmodelParams():
        return [
            assimilationAlg(solver, idealX, gt, postprocess, pretrainingParams)
            for _ in range(numberOfSubmodels)
        ]

    # This captures the text output, to prevent flooding the cell output with it
    # Doesn't work for pyABC.
    output = io.StringIO()
    with redirect_stdout(output):
        # pretraining results should be shared between all the execution paths
        # branching from this point
        submodelParams = amb("submodel-params", [once(computeSubmodelParams)])

    submodel_res = []
    for i in range(numberOfSubmodels):
        vars = submodelParams[i]
        res = problem.solver(*vars)
        submodel_res.append(res)

    # Let's compare ADAO and evolution
    assimilation_method = amb("train-alg", [runADAO, runEvolution])

    # Various supermodeling methods
    supermodel_type = amb("supermodel", supermodel_types)

    runner = supermodel_type(assimilation_method)
    with redirect_stdout(output):
        res = runner(Experiment(problem, newParams), submodelParams)

    # we can append some additional data here
    res["submodel-outputs"] = submodel_res
    res["log"] = output.getvalue()
    res["ground-truth"] = full_gt
    return res


# %% [markdown]
# ## Execution
#
# To actually execute a non-deterministic function (that is, to construct and explore
# the execution tree), we can use `supy.amb.nondeterministic`. It returns a generator
# object, that produces events occurring during the computation, like "execution path
# complete", or "node exploration started". Here we use the version using an
# asynchronous generator, so that we can post-process and report results during the
# computation, using Python's `asyncio` facilities.
#
# Here we execute all the paths of the experiment in parallel. Due to Python's GIL
# (Global Interpreter Lock), threads are not useful for parallelizing CPU intensive
# workloads, so we use multiple processes. This also helps to avoid issues with
# libraries not designed for multithreaded use.

# %%
set_seed(3)

computation = nondeterministic(
    run_stuff,
    mode="async",  # allow plotting etc. during computations
    node_events=True,
    execution="processes",  # parallel execution
    pass_exceptions=True,  # propagate exceptions during execution
    mp_context=multiprocessing.get_context("fork"),  # see below
)


# %% [markdown]
# ````{note} Using execution="processes" on non-linux platforms
# :class: attention
# Under the hood, parallel execution with `"processes"` uses functionality of Pyton's
# `multiprocessing` module. There are multiple ways to create new worker processes. As
# of Python 3.12, default method on Linux is `"fork"`, which forks the current Python
# interpreter and executes the assigned tasks in the child process. This means the task
# is executed in the same environment - child process has a copy of all the global
# variables, all the modules imported by the parent, all the functions and classes
# defined in the interpreter session etc. Using other methods (`"spawn"` and
# `"forkserver"`), workers start with a brand new interpreter session. If the task
# attempts to access global functions, classes or imported modules from a Jupyter
# notebook, it will fail, since these are not available in a fresh interpreter session.
#
# Since the `"fork"` method has [some
# issues](https://discuss.python.org/t/switching-default-multiprocessing-context-to-spawn-on-posix-as-well/21868),
# the general trend has been to move away from it in favor of `"spawn"`. On macOS
# `"spawn"` has been the default since Python 3.8, on Linux it will be, starting with
# 3.14. On Windows, it is the only available method.
#
# To make the code work with `"spawn"`, callable passed to `nondeterministic` must be
# possible to execute in an empty environment. In particular:
#
# - it must not access global variables
# - all the external names it uses must be imported inside the function
#
# If one wants to share some values between the experiment function and other parts
# of the code, once can add them as arguments, and use `functools.partial` to pass
# them to the experiment function:
# ```python
# data = ...
#
# def experiment(amb, some_data):
#     ...
#
# nondeterministic(functools.partial(experiment, some_data=data), ...)
# ```
# make the experiment function a class method:
# ```python
# class MyExperiment:
#     data = ...
#
#     def run(self, amb):
#         data = self.data
#         ...
#
# experiment = MyExperiment()
# nondeterministic(experiment.run, ...)
#
# data = experiment.data
# ...
# ```
# or define them inside the experiment function and include them as part of
# the return value.
#
# In the notebook code above, `"fork"` method is explicitly specified.
# ````

# %% [markdown]
# Next, we consume the events produced by the generator returned from the
# `nondeterministic` call and build a tree of execution paths, where nodes represent
# choices made by `amb` and leaves store values returned by the execution paths. The
# tree is built in the background - `execution_tree` returns immediately an initially
# empty tree, and new nodes are being added as the computation progresses.

# %%
results = execution_tree(computation)


# %% [markdown]
# ### (Optional) Saving/loading results
#
# At this point, we may want to save the solution tree to a file. As `DataTree`
# objects are picklable, it can be done using tools from Python standard library.
# Serialized data tree can then be loaded back in this notebook, or any other Python
# code. This is especially useful while developing post-processing code: instead of
# running the entire computation every time to test a small change, one can use a
# saved results tree, reload it here and use `Run Selected Cell and All Below` to
# execute the rest of the notebook from this point. This feature can also be used to
# persist the results outside of notebook, or even separate the process: generate
# results in one notebook, and analyze them in another.
#
# For these workflows to make sense, the serialized tree should be complete. While
# saving a partial tree during the building process is possible, it is probably not
# very useful, since there is currently no way to restart the experiment from its
# current state. Note that this requires waiting for all the computations to finish
# before processing the results, which precludes inspecting partial results for early
# feedback.


# %%
async def save(path):
    await results.wait_until_complete()  # wait for the computation to end
    with open(path, "wb") as stream:
        pickle.dump(results, stream)


def load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


# Save results to a file
# await save("results-file")

# Load saved results from file
# results = load("results-file")

# %% [markdown]
# ## Postprocessing
#
# The experiment function produces the raw results in a simple format. Often, we are
# interested in various derived metrics (errors, averages, comparisons etc.), and we
# want to present them in a graphical form. Postprocessing is conceptually divided
# into two stages:
#
# - *Decorating* the results tree, i.e. adding new node attributes
# - Reporting results, i.e. printing and/or plotting
#
# Since early feedback during an experiment is useful, these two stages are carried
# out in parallel with the computation itself. Each post-processing operation
# typically requires only a small part of the results tree to be available, and as
# soon as the required part is completed, the operation is initiated. This allows us
# to see the final form of the results immediately, without waiting for the entire
# experiment to finish.
#
# ```{note}
# There is quite a bit of "manual" work happening here. A lot of it can be generalized
# and extracted, making post-processing have more of a "putting together building
# blocks" kind of feel. Following code in the present state if more of an experiment
# and a proof-of-concept.
#
# In particular, mechanisms for node selection are sorely lacking. In simple cases
# presented here one can make do with explicit `if`s and node names. In the future,
# the plan is to adapt a subset of XPath, leveraging the structural similarity of
# `DataTree` and XML.
# ```


# %% [markdown]
# ### Decorating the results tree
#
# Nodes of the results tree correspond to the nodes of the decision tree. Each node
# (other than the root) has a name (string passed to the corresponding `amb` call)
# and a value (one of the possible values passed to `amb`). Leaf nodes also have a
# `"result"` attribute, which can be used to access the value returned by the
# experiment function at the end of the corresponding execution path. Errors during
# the execution are recorded in the result tree as well; see
# `supy.datatree.amb.execution_tree` documentation for details.
#
# New attributes can be freely added to the nodes of the tree. Attribute values are
# retrieved asynchronously:
# ```python
# val = await node["attribute-name"]
# ```
# The execution is suspended until the attribute is set. This allows multiple
# independent, modular tasks to work cooperatively on decorating the tree
# without explicit communication or barries.
#
# Iteration over nodes of the data tree produces nodes in post-order, and the nodes
# are guaranteed to be complete, in the sense that no new nodes will be added to
# the subtree rooted at it. This is enough to guarantee correct computation order
# if an attribute can be computed using its value at the node's children (*synthesized
# attributes*, in the attribute grammar paralance). However, in case of multiple
# attributes with complex interactions, manually managing resulting data dependencies
# can be tricky, especially in the context of asynchronous execution. It is advised
# to add attributes using the `decorate` method of the data tree, which takes a
# function and invokes it on all the nodes (current and future), each call being an
# independently scheduled task.
#
# ```{warning}
# When the value of a missing attribute is accessed, a pending request is recorded.
# When the attribute is added later, tasks waiting for its value are awakened.
# If the attribute is never added (e.g. its name has been misspelled), tasks will
# wait indefinitely. There is currently no mechanism to detect this.
# ```


# %% [markdown]
# #### Error computation
#
# Here we compute the maximum and `l2` errors of the solutions obtained in each
# computation with respect to the ground truth. We store these errors as an
# attribute of the data tree leafs, so that other parts of post-processing phase
# can access them.


# %%
@results.decorate()
async def compute_errors(node):
    # 'async' turns the function into a coroutine

    # select leaves - all the paths end with "supermodel" choice
    if node.name == "supermodel":
        # "result" is certainly available already, but we need
        # the 'await', since the attribute retrieval is asynchronous
        res = await node["result"]

        # What we get from this 'await' is a normal value,
        # no other asynchronous operations involved.
        gt = res["ground-truth"]
        diff = res["data"] - gt
        errors = {}

        errors["max"] = np.abs(diff).max()
        errors["max-rel"] = errors["max"] / np.abs(gt).max()

        errors["l2"] = np.linalg.norm(diff)
        errors["l2-rel"] = errors["l2"] / np.linalg.norm(gt)

        node["errors"] = errors


# %% [markdown]
# #### Aggregating results from children
#
# Next, we may want to aggregate the results of multiple single experiments to
# produce some summary reports. To this end, we gather relevant data from the
# descendants of the nodes at the level where we want to summarize. In this case,
# we chose to produce summary results for each iteration, where we compare all the
# types of supermodeling, and the training algorithms used. We also include results
# produced by the submodels shared by all these computations.
#
# This step is a bit more manual and involved, though the reason is mostly lack of
# necessary infrastructure, rather than some inherent complexity.


# %%
async def path_up(node, up_to=None):
    items = []
    while node is not up_to:
        # root does not have a value
        if node.parent is not None:
            items.append((node.name, await node["value"]))
        node = node.parent
    return tuple(reversed(items))


def path_to_label(path_vals):
    return path_vals["supermodel"].__name__, path_vals["train-alg"].__name__


@results.decorate()
async def aggregate(node):
    if node.name == "submodel-params":
        t0, t1 = modelingWindow
        xs = list(range(t0, t1 + 1))

        # one dict to gather results
        aggr = {}

        # grab all the leaves below
        leaves = [n for n in node.descendants if n.name == "supermodel"]

        # submodels are the same
        first_res = await leaves[0]["result"]

        aggr["submodels"] = pd.DataFrame(
            {i: sub["data"] for i, sub in enumerate(first_res["submodel-outputs"])},
            index=xs,
        )

        aggr["ground-truth"] = pd.DataFrame(first_res["ground-truth"], index=xs)

        cols = {}
        errors = {}
        for leaf in leaves:
            path = await path_up(leaf, up_to=node)
            path_vals = dict(path)
            label = path_to_label(path_vals)

            res = await leaf["result"]
            # inversed order for easier plots
            # should not be done here
            cols[tuple(reversed(label))] = res["data"]
            errors[label] = await leaf["errors"]

        aggr["supermodels"] = pd.DataFrame(cols, index=xs)
        aggr["errors"] = pd.DataFrame(errors)

        node["aggregate"] = aggr


# %% [markdown]
# ### Reporting


# %% [markdown]
# This is a piece of general infrastructure that will later be moved to the
# library part.


# %%
class OutContext:
    def __init__(self, box, title):
        self.out = widgets.Output()
        self.box = box
        self.title = title

    def __enter__(self):
        return self.out

    def __exit__(self, *args):
        container = widgets.Accordion(children=[self.out], titles=(self.title,))
        self.box.children += (container,)


class OutputBox:
    def __init__(self):
        self.box = widgets.VBox()

    def page(self, title):
        return OutContext(self.box, title)

    def display(self):
        display(self.box)


def run_with_output(desc):
    def do(f):
        box = OutputBox()
        box.display()
        spawn_task(f(box), desc)
        return f

    return do


# %% [markdown]
# #### Reporting aggregate results
#
# Here we report results from nodes higher up in the execution tree, i.e. aggregate
# results of multiple descendant nodes. Here we display the results produced by the
# decoration process at each iteration node.


# %%
def format_path(path):
    items = []
    for name, value in path:
        if name != "submodel-params":
            val_str = value.__name__ if hasattr(value, "__name__") else str(value)
            items.append(f"{name}: {val_str}")
    return " | ".join(items)


# %%
async def output_report(node, out):
    aggregate = await node["aggregate"]

    # Print errors
    df = aggregate["errors"].transpose()
    for col in df:
        # relative errors as percents
        if "rel" in col:
            df[col] = df[col].map("{:.2%}".format)
    out.append_display_data(df)

    ground_truth = aggregate["ground-truth"]
    xs = ground_truth.index

    # Plot supermodel results
    supermodels = aggregate["supermodels"]
    algs = supermodels.columns.get_level_values(0).unique()

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    fig.suptitle("Supermodel results")

    for i, alg in enumerate(algs):
        ax = axs[i]
        ax.set_title(f"Trained with {alg}")
        ax.plot(xs, ground_truth, linestyle="--", label="ground truth")

        for method in supermodels[alg]:
            ax.plot(xs, supermodels[alg, method], label=method)

        for x in learningWindow:
            ax.axvline(x, linestyle="--", color="black")
        ax.legend()

    fig.tight_layout()
    plt.close(fig)
    out.append_display_data(fig)

    # Plot submodels used
    submodels = aggregate["submodels"]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_title("Submodels")
    ax.plot(xs, ground_truth, linestyle="--", label="ground truth")
    for i in submodels:
        ax.plot(xs, submodels[i], label=f"submodel {i}")

    for x in learningWindow:
        ax.axvline(x, linestyle="--", color="black")
    ax.legend()

    fig.tight_layout()
    plt.close(fig)
    out.append_display_data(fig)


@run_with_output("Report aggregate results")
async def report(cell):
    async for node in results:
        if node.name == "submodel-params":
            path = await path_up(node)
            with cell.page(format_path(path)) as out:
                await output_report(node, out)


# %% [markdown]
# #### Reporting results per-execution-path
#
# Here we display detailed reports (plots, errors etc.) for each execution path.
# New reports appears right after a single execution path is completed.
# Each execution path has its own collapsible box, that can be opened to display
# the details.
#
# ```{note}
# This is actually surprisingly slow, apparently due to `append_stdout` and
# `append_display_data`. Disabling the following cell halves the execution
# time of the entire notebook. That is clearly an issue with `ipywidgets.Output`,
# since simply printing and displaying `matplotlib` figures directly in the
# notebook is much faster. Unfortunately, there seems to be no any other way to
# preserve the text and plot output order, or even prevent the figures from
# appearing in wrong cells altogether.
#
# Comment out the `@run_with_output` line to disable this part. Since the results
# are saved in the data tree, one can always uncomment it and re-run this cell
# at a later time, should detailed plots prove desirable.
# ```


# %%
@run_with_output("Reports for each leaf")
async def report_leaves(cell):
    async for node in results:
        # select leaves - all the paths end with "supermodel" choice
        if node.name == "supermodel":
            path = await path_up(node)
            path_vals = dict(path)

            with cell.page(format_path(path)) as out:
                res = await node["result"]
                out.append_stdout("Result keys:\n")
                for key, val in res.items():
                    out.append_stdout(f"  {key}: {type(val)}\n")

                out.append_stdout(f"States: {res['states'].shape}\n")

                # Print errors computed earlier
                errors = await node["errors"]
                # convert it to DataFrame to get nice output
                df = pd.DataFrame({norm: [val] for norm, val in errors.items()})
                for col in df:
                    # relative errors as percents
                    if "rel" in col:
                        df[col] = df[col].map("{:.2%}".format)
                out.append_display_data(df)

                # We can do any custom plots we want here
                t0, t1 = modelingWindow
                xs = list(range(t0, t1 + 1))
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(xs, res["data"], label="result")
                ax.plot(xs, res["ground-truth"], linestyle="--", label="ground truth")
                for x in learningWindow:
                    ax.axvline(x, linestyle="--", color="black")
                ax.legend()
                fig.tight_layout()
                plt.close(fig)
                out.append_display_data(fig)

                submodels = res["submodel-outputs"]
                fig, ax = plt.subplots(figsize=(5, 3))

                for i, sub in enumerate(submodels):
                    ax.plot(xs, sub["data"], label=f"submodel {i}")

                ax.plot(xs, res["ground-truth"], linestyle="--", label="ground truth")
                for x in learningWindow:
                    ax.axvline(x, linestyle="--", color="black")

                ax.legend()
                fig.tight_layout()
                plt.close(fig)
                out.append_display_data(fig)

                # Method-specific plots
                if path_vals["supermodel"] == CPTSuperModelRunner:
                    weights = pd.DataFrame(res["weights"])
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.set_title("CPT weights")

                    for i in weights:
                        ax.plot(weights[i], label=f"w{i}")
                    ax.legend()
                    fig.tight_layout()
                    plt.close(fig)
                    out.append_display_data(fig)


# %% [markdown]
# ## Done
#
# Here we just wait for the computation to end. Post-processing may still be running.

# %%
await results.wait_until_complete()
print("All done")
