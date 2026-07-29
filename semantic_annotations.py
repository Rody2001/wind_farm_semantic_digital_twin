"""
semantic_annotations.py
===================================================================
Semantic annotations for the wind farm digital twin.

Two kinds live here:

* Structural annotations (Tower, Nacelle, Hub, RotorBlades, TowerBase) say what
  a Body IS. They carry no data of their own -- the geometry sits on the Body.

* Measured annotations carry a scalar VALUE that changes while MuJoCo runs.
  SemanticWindDriver.step() updates them every tick, in the same loop that moves
  the turbines, and every query reads them instead of reaching into the driver.

      environment:  WindSpeed, WindDirection, Temperature
      per turbine:  RotorSpeed      (rooted at the hub -- the body that turns)
                    NacelleYaw      (rooted at the nacelle -- the body that yaws)
                    GeneratedPower  (rooted at the hub)
                    GeneratedEnergy (rooted at the hub)

The per-turbine ones also carry `turbine`, the farm-prefixed name from
combined_specs(), so a query can ask for "Farm_East_tall" by name.
===================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body


class ScalarValueMixin:
    """Behaviour shared by every measured annotation.

    Deliberately NOT a dataclass and carrying no fields of its own, so it can be
    mixed into both SemanticAnnotation and HasRootBody subclasses without
    disturbing dataclass field order anywhere in the MRO.
    """

    def set(self, value, stamp: float = None) -> float:
        """Write a new measurement. Called by the driver, never by a query."""
        self.value = float(value)
        self.stamp = time.time() if stamp is None else float(stamp)
        return self.value

    def is_fresh(self, max_age_s: float = 2.0) -> bool:
        """False if nothing has written to this annotation recently.

        Lets a query tell "the rotor really is stopped" apart from "nothing has
        been updating this world".
        """
        return self.stamp > 0.0 and (time.time() - self.stamp) <= max_age_s

    def __str__(self) -> str:
        who = getattr(self, "turbine", "")
        who = f"{who}." if who else ""
        return f"{who}{type(self).__name__}={self.value:.3f} {self.unit}".strip()


# ------------------------------------------------------------------ #
# structural annotations (unchanged)
# ------------------------------------------------------------------ #
@dataclass(eq=False)
class Tower(HasRootBody):
    ...


@dataclass(eq=False)
class RotorBlades(HasRootBody):
    ...


@dataclass(eq=False)
class TowerBase(HasRootBody):
    ...


@dataclass(eq=False)
class Nacelle(HasRootBody):
    ...


@dataclass(eq=False)
class Hub(HasRootBody):
    ...


# ------------------------------------------------------------------ #
# measured environment annotations -- properties of the world itself
# ------------------------------------------------------------------ #
@dataclass(eq=False)
class MeasuredScalar(SemanticAnnotation, ScalarValueMixin):
    """A scalar property of the world that the simulation keeps up to date."""
    value: float = field(default=0.0, kw_only=True)
    unit: str = field(default="", kw_only=True)
    stamp: float = field(default=0.0, kw_only=True)   # wall clock of the last write


@dataclass(eq=False)
class WindSpeed(MeasuredScalar):
    """Free-stream wind speed at hub height."""
    unit: str = field(default="m/s", kw_only=True)


@dataclass(eq=False)
class WindDirection(MeasuredScalar):
    """Compass bearing the wind comes FROM: 0 = north, 90 = east, 270 = west."""
    unit: str = field(default="deg", kw_only=True)


@dataclass(eq=False)
class Temperature(MeasuredScalar):
    """Air temperature, which sets the air density through the ideal gas law."""
    unit: str = field(default="degC", kw_only=True)


# ------------------------------------------------------------------ #
# measured per-turbine annotations -- properties of a specific Body
# ------------------------------------------------------------------ #
@dataclass(eq=False)
class MeasuredBodyScalar(HasRootBody, ScalarValueMixin):
    """A scalar property of one Body, belonging to one named turbine."""
    turbine: str = field(default="", kw_only=True)
    value: float = field(default=0.0, kw_only=True)
    unit: str = field(default="", kw_only=True)
    stamp: float = field(default=0.0, kw_only=True)


@dataclass(eq=False)
class RotorSpeed(MeasuredBodyScalar):
    """Rotational speed of the hub -- the body that actually turns."""
    unit: str = field(default="rpm", kw_only=True)


@dataclass(eq=False)
class NacelleYaw(MeasuredBodyScalar):
    """Yaw angle of the nacelle, local to the tower, in degrees."""
    unit: str = field(default="deg", kw_only=True)


@dataclass(eq=False)
class GeneratedPower(MeasuredBodyScalar):
    """Power this turbine is generating right now."""
    unit: str = field(default="W", kw_only=True)


@dataclass(eq=False)
class GeneratedEnergy(MeasuredBodyScalar):
    """Energy this turbine has generated since the run started."""
    unit: str = field(default="kWh", kw_only=True)