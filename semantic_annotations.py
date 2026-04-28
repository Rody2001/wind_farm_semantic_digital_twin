from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body



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