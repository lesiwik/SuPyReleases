from dataclasses import dataclass
from typing import Annotated, ClassVar

from supy.meta.annotations import Param, Var, event, model
from supy.ode import ODE
from supy.problem import Event


@event(label="Therapy application")
class Therapy(Event):
    strength: Annotated[float, Param()]


@model(label="Ribba tumor model")
@dataclass
class TumorRibba(ODE):
    lambdap: Annotated[float, Param()]
    K: Annotated[float, Param()]
    kqpp: Annotated[float, Param()]
    kpq: Annotated[float, Param()]
    gammap: Annotated[float, Param()]
    gammaq: Annotated[float, Param()]
    deltaqp: Annotated[float, Param()]
    KDE: Annotated[float, Param()]

    state_vars: ClassVar = ("C", "P", "Q", "QP")
    derived_vars: ClassVar = ("tumor_diameter",)
    events: ClassVar = (Therapy,)

    output: ClassVar = (
        Var("C"),
        Var("P"),
        Var("Q"),
        Var("QP"),
        Var("tumor_diameter", label="Diameter", desc="Diameter of the tumor"),
    )

    def rhs(self, t, state):
        C, P, Q, QP = state
        Pstar = P + Q + QP
        dCdt = -self.KDE * C
        dPdt = (
            self.lambdap * P * (1 - Pstar / self.K)
            + self.kqpp * QP
            - self.kpq * P
            - self.gammap * C * self.KDE * P
        )
        dQdt = self.kpq * P - self.gammaq * C * self.KDE * Q
        dQpdt = self.gammaq * C * self.KDE * Q - self.kqpp * QP - self.deltaqp * QP
        return [dCdt, dPdt, dQdt, dQpdt]

    def derived(self, state):
        C, P, Q, QP = state
        return [P + Q + QP]

    def handle_event(self, event, state) -> None:
        match event:
            case Therapy() as e:
                state[0] = e.strength
