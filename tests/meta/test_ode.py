from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest

import supy.meta as meta
from supy.meta.annotations import Group, Param, Var, event, model
from supy.ode import ODE
from supy.utils import Range

# There is a lot of type-annotations here inside untyped functions,
# but these annotations are for the purpose of testing extraction
# code only.
# mypy: disable-error-code = annotation-unchecked


def test_extracts_model_info():
    @model("Simple model", desc="For testing")
    @dataclass
    class Simple(ODE):
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()
    assert info.name == "Simple"
    assert info.label == "Simple model"
    assert info.desc == "For testing"


def test_extracts_params():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithParam(ODE):
        foo: Annotated[float, Param("FOO", desc="This is a foo")]
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()

    foo_param = meta.Param(
        name="foo",
        label="FOO",
        desc="This is a foo",
        type=float,
        required=True,
        range=Range.UNKNOWN,
    )
    assert info.params == (foo_param,)


def test_extracts_params_with_range():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithParam(ODE):
        foo: Annotated[float, Param("FOO", desc="This is a foo", range=Range(1, 3))]
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()

    foo_param = meta.Param(
        name="foo",
        label="FOO",
        desc="This is a foo",
        type=float,
        required=True,
        range=Range(min=1, max=3),
    )
    assert info.params == (foo_param,)


def test_extracts_params_with_partial_range():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithParam(ODE):
        foo: Annotated[float, Param("FOO", desc="This is a foo", range=Range(max=3))]
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()

    foo_param = meta.Param(
        name="foo",
        label="FOO",
        desc="This is a foo",
        type=float,
        required=True,
        range=Range(min=None, max=3),
    )
    assert info.params == (foo_param,)


def test_extracts_optional_param():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithParam(ODE):
        foo: Annotated[float, Param("FOO", desc="This is a foo")] = 3.14
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()

    foo_param = meta.Param(
        name="foo",
        label="FOO",
        desc="This is a foo",
        type=float,
        required=False,
        range=Range.UNKNOWN,
    )
    assert info.params == (foo_param,)


def test_output_using_state_vars():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithState(ODE):
        state_vars: ClassVar = ("foo", "bar")
        derived_vars: ClassVar = ()
        events: ClassVar = ()

        output: ClassVar = (
            Var("foo", label="Foo", desc="Foo density"),
            Var("bar", label="Bar", desc="Local bar-ity"),
        )

    (info,) = meta.models.all()

    foo_var = meta.OutputVar(
        name="foo",
        label="Foo",
        desc="Foo density",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    bar_var = meta.OutputVar(
        name="bar",
        label="Bar",
        desc="Local bar-ity",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    assert info.output == (foo_var, bar_var)


def test_output_using_state_vars_with_range():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithState(ODE):
        state_vars: ClassVar = ("foo",)
        derived_vars: ClassVar = ()
        events: ClassVar = ()

        output: ClassVar = (
            Var("foo", label="Foo", desc="Foo density", range=Range(1, 3)),
        )

    (info,) = meta.models.all()

    foo_var = meta.OutputVar(
        name="foo",
        label="Foo",
        desc="Foo density",
        type=meta.TimeSeries,
        range=Range(min=1, max=3),
    )
    assert info.output == (foo_var,)


def test_rejects_dynamic_state_vars():
    with pytest.raises(TypeError, match="'state_vars' must be .* class attribute"):

        @model("Simple model", desc="For testing")
        @dataclass
        class WithState(ODE):
            derived_vars: ClassVar = ()
            output: ClassVar = ()
            events: ClassVar = ()

            @property
            def state_vars(self):
                return ()


def test_output_using_derived_vars():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithOutput(ODE):
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ("foo", "bar")
        events: ClassVar = ()

        output: ClassVar = (
            Var("foo", label="Foo", desc="Foo density"),
            Var("bar", label="Bar", desc="Local bar-ity"),
        )

    (info,) = meta.models.all()

    foo_var = meta.OutputVar(
        name="foo",
        label="Foo",
        desc="Foo density",
        type=meta.TimeSeries,
        range=Range(None, None),
    )
    bar_var = meta.OutputVar(
        name="bar",
        label="Bar",
        desc="Local bar-ity",
        type=meta.TimeSeries,
        range=Range(None, None),
    )
    assert info.output == (foo_var, bar_var)


def test_output_using_derived_vars_with_range():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithDerived(ODE):
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ("foo",)
        events: ClassVar = ()

        output: ClassVar = (
            Var("foo", label="Foo", desc="Foo density", range=Range(1, 3)),
        )

    (info,) = meta.models.all()

    foo_var = meta.OutputVar(
        name="foo",
        label="Foo",
        desc="Foo density",
        type=meta.TimeSeries,
        range=Range(min=1, max=3),
    )
    assert info.output == (foo_var,)


def test_output_with_groups():
    @model("Simple model", desc="For testing")
    @dataclass
    class WithGroups(ODE):
        state_vars: ClassVar = ("asdf", "qwer")
        derived_vars: ClassVar = ("foo", "bar")
        events: ClassVar = ()

        output: ClassVar = (
            Group(
                "cool",
                label="cool variables",
                desc="The variables which are cool",
                items=(
                    Var("foo", label="Foo", desc="Foo density"),
                    Group(
                        "best",
                        label="most cool variables",
                        items=(
                            Var("bar", label="Bar", desc="Local bar-ity"),
                            Var("qwer", label="QWER", desc="qwertyness"),
                        ),
                    ),
                ),
            ),
            Var("asdf", label="ASDF", desc="asdf quantity"),
        )

    (info,) = meta.models.all()

    foo_var = meta.OutputVar(
        name="foo",
        label="Foo",
        desc="Foo density",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    bar_var = meta.OutputVar(
        name="bar",
        label="Bar",
        desc="Local bar-ity",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    qwer_var = meta.OutputVar(
        name="qwer",
        label="QWER",
        desc="qwertyness",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    asdf_var = meta.OutputVar(
        name="asdf",
        label="ASDF",
        desc="asdf quantity",
        type=meta.TimeSeries,
        range=Range.UNKNOWN,
    )
    assert info.output == (
        meta.OutputGroup(
            name="cool",
            label="cool variables",
            desc="The variables which are cool",
            items=(
                foo_var,
                meta.OutputGroup(
                    name="best",
                    label="most cool variables",
                    desc=None,
                    items=(bar_var, qwer_var),
                ),
            ),
        ),
        asdf_var,
    )


def test_rejects_dynamic_derived_vars():
    with pytest.raises(TypeError, match="'derived_vars' must be .* class attribute"):

        @model("Simple model", desc="For testing")
        @dataclass
        class WithDynamicDerived(ODE):
            state_vars: ClassVar = ()
            output: ClassVar = ()
            events: ClassVar = ()

            @property
            def derived_vars(self):
                return ()


def test_rejects_non_time_series_output():
    with pytest.raises(TypeError, match="must be TimeSeries"):

        @model("Simple model", desc="For testing")
        @dataclass
        class WithDynamicDerived(ODE):
            state_vars: ClassVar = ("foo",)
            derived_vars: ClassVar = ()
            events: ClassVar = ()
            output: ClassVar = (Var("foo", label="Foo", desc="Foo density", type=int),)


def test_output_vars_must_be_state_or_derived():
    with pytest.raises(ValueError, match="Unknown variable 'foo'"):

        @model("Simple model", desc="For testing")
        @dataclass
        class InvalidOutput(ODE):
            state_vars: ClassVar = ()
            derived_vars: ClassVar = ()
            events: ClassVar = ()
            output: ClassVar = (Var("foo", label="Foo", desc="Foo density"),)


def test_extracts_events():
    @event(label="Some event")
    class SomeEvent:
        foo: Annotated[bool, Param("Foobility")]

    @model("Simple model", desc="For testing")
    @dataclass
    class WithEvents(ODE):
        state_vars: ClassVar = ("foo",)
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = (SomeEvent,)

    (info,) = meta.models.all()

    assert info.events == (meta.events["SomeEvent"],)


def test_reformats_descriptions():
    @model(
        "Simple model",
        desc="""
                This is
                an example
                  description.

                ASDF!
                """,
    )
    @dataclass
    class WithOutput(ODE):
        state_vars: ClassVar = ()
        derived_vars: ClassVar = ()
        output: ClassVar = ()
        events: ClassVar = ()

    (info,) = meta.models.all()

    assert info.desc == "This is\nan example\n  description.\n\nASDF!"


def test_model_extraction(mocker):
    @model(
        label="Lotka-Volterra predator-prey",
        desc="""
        First-order nonlinear differential equations used to describe
        dynamics of biological systems with two interacting species.
        """,
    )
    @dataclass
    class LotkaVolterra(ODE):
        a: Annotated[
            float, Param(r"\alpha", desc="Maximum prey per capita growth rate")
        ]
        b: Annotated[
            float,
            Param(
                r"\beta",
                desc="The effect of the presence of predators on the prey growth rate",
            ),
        ]
        c: Annotated[
            float,
            Param(
                r"\gamma",
                desc="The effect of the presence of pery on the predator's growth rate",
            ),
        ]
        d: Annotated[float, Param(r"\delta", desc="Predator's per capita death rate")]

        state_vars: ClassVar = ("x", "y")

        derived_vars: ClassVar = ("total",)

        output: ClassVar = (
            Var("x", label="prey", desc="Prey population density"),
            Var("y", label="predators", desc="Predator population density"),
            Var(
                "total",
                label="total",
                desc="Total population size (sum of prey and predators)",
            ),
        )

        events: ClassVar = ()

        def rhs(self, t: float, state) -> list[float]:
            x, y = state
            dx = (self.a - self.b * y) * x
            dy = (self.c * x - self.d) * y
            return [dx, dy]

        def derived(self, state) -> list[float]:
            x, y = state
            return [x + y]

    (info,) = meta.models.all()

    ANY = mocker.ANY
    expected = meta.Model(
        name="LotkaVolterra",
        label="Lotka-Volterra predator-prey",
        desc="First-order nonlinear differential equations used to describe\n"
        "dynamics of biological systems with two interacting species.",
        params=(
            meta.Param(
                name="a",
                label=r"\alpha",
                desc="Maximum prey per capita growth rate",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
            meta.Param(
                name="b",
                label=r"\beta",
                desc="The effect of the presence of predators on the prey growth rate",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
            meta.Param(
                name="c",
                label=r"\gamma",
                desc="The effect of the presence of pery on the predator's growth rate",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
            meta.Param(
                name="d",
                label=r"\delta",
                desc="Predator's per capita death rate",
                type=float,
                required=True,
                range=Range.UNKNOWN,
            ),
        ),
        events=(),
        output=(
            meta.OutputVar(
                name="x",
                label="prey",
                desc="Prey population density",
                type=meta.TimeSeries,
                range=Range.UNKNOWN,
            ),
            meta.OutputVar(
                name="y",
                label="predators",
                desc="Predator population density",
                type=meta.TimeSeries,
                range=Range.UNKNOWN,
            ),
            meta.OutputVar(
                name="total",
                label="total",
                desc="Total population size (sum of prey and predators)",
                type=meta.TimeSeries,
                range=Range.UNKNOWN,
            ),
        ),
        new_instance=ANY,
    )

    assert info == expected
