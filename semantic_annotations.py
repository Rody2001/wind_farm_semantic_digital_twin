"""
semantic_annotations.py
===================================================================
Semantic annotations for the wind farm digital twin.

Two kinds live here:

* Structural annotations (Tower, Nacelle, Hub, RotorBlades, TowerBase) say what
  a Body IS and what it is made of. The geometry sits on the Body; the material
  is a property only the semantic model records, since MuJoCo's own materials
  describe nothing but how a surface is drawn.

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
# structural annotations: what a body IS, and what it is made of
# ------------------------------------------------------------------ #
@dataclass(eq=False)
class TurbinePart(HasRootBody):
    """A structural part of a turbine.

    Every part carries the material it is made of, so the world can be asked
    what a turbine is built from and not only what shape it has. The defaults
    below are the materials used in a modern three-bladed turbine; pass
    `material=` to override one.

    Querying every part of every turbine at once is then just::

        for part in world.get_semantic_annotations_by_type(TurbinePart):
            print(part.name, part.material)
    """
    material: str = field(default="", kw_only=True)


@dataclass(eq=False)
class Tower(TurbinePart):
    """The tube the nacelle stands on. Usually rolled and welded steel sections,
    sometimes concrete in the lower part of very tall towers."""
    material: str = field(default="steel", kw_only=True)


@dataclass(eq=False)
class RotorBlades(TurbinePart):
    """A rotor blade: a glass-fibre reinforced epoxy shell over a load-carrying
    spar, which in longer blades is reinforced with carbon fibre."""
    material: str = field(default="glass fibre reinforced polymer", kw_only=True)


@dataclass(eq=False)
class TowerBase(TurbinePart):
    """The foundation the tower is anchored into."""
    material: str = field(default="reinforced concrete", kw_only=True)


@dataclass(eq=False)
class Nacelle(TurbinePart):
    """The housing around the drive train. The structure inside is steel; the
    weatherproof shell itself is a moulded composite."""
    material: str = field(default="glass fibre reinforced polymer", kw_only=True)


@dataclass(eq=False)
class Hub(TurbinePart):
    """The casting the blades are bolted to, carrying the whole rotor load."""
    material: str = field(default="cast iron", kw_only=True)


# The keys accepted by WindTurbine.create_with_new_body_in_world(materials=...).
MATERIAL_KEYS = ("tower_base", "tower", "nacelle", "hub", "rotor_blade")


def find_part(world, part_name: str):
    """The TurbinePart annotation named `part_name`, e.g. "Farm_Big_1_tower".

    Matches on the bare name rather than the prefixed form, so "Farm_Big_1_tower"
    finds PrefixedName("None/Farm_Big_1_tower"). Returns None if there is no such
    part.
    """
    for part in world.get_semantic_annotations_by_type(TurbinePart):
        name = getattr(part, "name", None)
        if name is None:
            continue
        if getattr(name, "name", None) == part_name or str(name).endswith("/" + part_name):
            return part
    return None


def part_material(world, part_name: str) -> str:
    """What the named part is made of. Raises KeyError if there is no such part."""
    part = find_part(world, part_name)
    if part is None:
        raise KeyError(f"no turbine part named {part_name!r} in the world")
    return part.material


def bill_of_materials(world) -> dict:
    """Every turbine part in the world mapped to the material it is made of."""
    return {getattr(p.name, "name", str(p.name)): p.material
            for p in world.get_semantic_annotations_by_type(TurbinePart)}


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


# ------------------------------------------------------------------ #
# rating -- set once at creation, not updated by step()
# ------------------------------------------------------------------ #
@dataclass(eq=False)
class RatedPower(MeasuredBodyScalar):
    """Rated output of this turbine: the ceiling GeneratedPower is held at once
    the wind reaches RatedWindSpeed. Derived from blade length as
    L * (2240 / 69) unless the turbine was created with an explicit max_kw.

    Written once by WindTurbine.create_with_new_body_in_world(max_kw=...) --
    unlike the measured annotations above, nothing rewrites it every tick.
    """
    unit: str = field(default="kW", kw_only=True)


@dataclass(eq=False)
class RatedWindSpeed(MeasuredBodyScalar):
    """Wind speed at and above which this turbine holds RatedPower instead of
    following the cp curve: blade length * (12.5 / 69)."""
    unit: str = field(default="m/s", kw_only=True)


@dataclass(eq=False)
class CutOutWindSpeed(MeasuredBodyScalar):
    """Wind speed at and above which this turbine shuts down -- rotor stopped,
    no power at all: blade length * (28 / 69)."""
    unit: str = field(default="m/s", kw_only=True)