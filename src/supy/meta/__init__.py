"""
Data structures describing registered SuPy entities.

.. currentmodule:: supy.meta

This module provides facilities that allow registering, listing and lookup of SuPy
entities, such as:

- model definitions
- event classes
- supermodeling methods
- assimilation algorithms

These actions can be accomplished using the global `Register` object corresponding to
the type of entity (see the list below).

.. note::
    In order for the entity to be known by the system and appear in the register, it
    needs to be registered. This requires the module defining the entity to be loaded.

Discovering entities
====================

.. autosummary::
    :toctree: generated/

    models
    events
    supermodels
    assimilation_algorithms
    Register


Entity description
==================

Top-level entities:

.. autosummary::
    :toctree: generated/

    Model
    SuperModel
    AssimilationAlgorithm

Auxiliary entities:

.. autosummary::
    :toctree: generated/

    Entity
    Param
    EventType
    OutputVar
    OutputGroup

Type aliases:

.. autosummary::
    :toctree: generated/

    OutputItem
    TimeSeries
"""
from supy.meta._data import (
    AssimilationAlgorithm,
    Entity,
    EventType,
    Model,
    OutputGroup,
    OutputItem,
    OutputVar,
    Param,
    SuperModel,
    TimeSeries,
)
from supy.meta._register import (
    Register,
    assimilation_algorithms,
    events,
    models,
    supermodels,
)

__all__ = [
    "AssimilationAlgorithm",
    "Entity",
    "EventType",
    "Model",
    "OutputGroup",
    "OutputItem",
    "OutputVar",
    "Param",
    "Register",
    "SuperModel",
    "TimeSeries",
    "assimilation_algorithms",
    "events",
    "models",
    "supermodels",
]
