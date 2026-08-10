"""
semantic_environment.py
===================================================================
The measured annotations, kept in sync with MuJoCo.

The turbines in the semantic world move because SemanticWindDriver.step() writes
into world.state every tick. The measured annotations work the same way: the
same step() writes the current wind, temperature, rotor speeds, yaw angles and
power into SemanticAnnotation objects that live in the World.

    wind_turbine_sim.py (MuJoCo viewer / browser panel)
        -> wind_state file  +  /wind_farm/* ROS topics
            -> SemanticWindDriver.step()          (the only writer)
                -> annotations in the World
                    -> queries.py                  (readers)

Setup in main1.py:

    with world.modify_world():
        ...
        environment = EnvironmentAnnotations(world)      # the three environment ones
        annotate_turbines(world, turbine_runtimes)       # four per turbine

    node = rclpy.create_node("semantic_digital_twin")
    environment.attach_ros(node)                         # temperature subscription
===================================================================
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from semantic_annotations import (
    GeneratedEnergy, GeneratedPower, NacelleYaw, RotorSpeed,
    Temperature, WindDirection, WindSpeed,
)

TOPIC_PREFIX = "/wind_farm"
DEFAULT_TEMP_C = 15.0          # matches rho = 1.225 kg/m^3, same default as the sim


# ------------------------------------------------------------------ #
# per-turbine annotations
# ------------------------------------------------------------------ #
def annotate_turbines(world, turbine_runtimes: Dict) -> Dict[str, Dict]:
    """Create the four measured annotations for every turbine and register them.

    Must be called inside `with world.modify_world():`. RotorSpeed, GeneratedPower
    and GeneratedEnergy are rooted at the hub, because that is the body that turns
    and where the rotor extracts the power; NacelleYaw is rooted at the nacelle,
    the body that yaws.

    The annotations are stored back on each TurbineRuntime, so step() can update
    them without looking them up again every tick.
    """
    created = {}
    for name, runtime in turbine_runtimes.items():
        hub_body = runtime.hub_conn.child
        nacelle_body = runtime.nacelle_conn.child
        rpm_ann = RotorSpeed(root=hub_body, turbine=name)
        yaw_ann = NacelleYaw(root=nacelle_body, turbine=name)
        pow_ann = GeneratedPower(root=hub_body, turbine=name)
        nrg_ann = GeneratedEnergy(root=hub_body, turbine=name)
        for ann in (rpm_ann, yaw_ann, pow_ann, nrg_ann):
            world.add_semantic_annotation(ann)
        runtime.rotor_speed_ann = rpm_ann
        runtime.nacelle_yaw_ann = yaw_ann
        runtime.power_ann = pow_ann
        runtime.energy_ann = nrg_ann
        created[name] = {"rpm": rpm_ann, "yaw": yaw_ann,
                         "power": pow_ann, "energy": nrg_ann}
    return created


# ------------------------------------------------------------------ #
# the three environment annotations
# ------------------------------------------------------------------ #
class EnvironmentAnnotations:
    """Creates the three environment annotations, registers them, keeps them fed.

    Construct inside `with world.modify_world():` so they are registered along
    with the rest of the world.
    """

    def __init__(self, world, temperature_c: float = DEFAULT_TEMP_C,
                 register: bool = True):
        self.world = world
        self.wind_speed_ann = WindSpeed()
        self.wind_direction_ann = WindDirection()
        self.temperature_ann = Temperature(value=temperature_c)
        self.annotations = [self.wind_speed_ann, self.wind_direction_ann,
                            self.temperature_ann]
        if register:
            for ann in self.annotations:
                world.add_semantic_annotation(ann)
        self._subscriptions = []

    # ---- the update path, called from SemanticWindDriver.step() ------ #
    def update(self, wind_speed: float = None, wind_direction_deg: float = None,
               temperature_c: float = None) -> None:
        """Write the current environment into the annotations. Cheap; call every step."""
        if wind_speed is not None:
            self.wind_speed_ann.set(wind_speed)
        if wind_direction_deg is not None:
            self.wind_direction_ann.set(wind_direction_deg % 360.0)
        if temperature_c is not None:
            self.temperature_ann.set(temperature_c)

    # ---- temperature comes over ROS, since no file carries it -------- #
    def attach_ros(self, node, prefix: str = TOPIC_PREFIX) -> None:
        """Subscribe to /wind_farm/temperature_c so the third annotation follows too.

        Wind speed and direction are also published, but the driver already reads
        them from the wind-state file, so subscribing would duplicate work.
        """
        from std_msgs.msg import Float64

        self._subscriptions.append(
            node.create_subscription(
                Float64, f"{prefix}/temperature_c",
                lambda msg: self.temperature_ann.set(msg.data), 10))

    # ---- read side --------------------------------------------------- #
    @property
    def wind_speed(self) -> float:
        return float(self.wind_speed_ann.value)

    @property
    def wind_direction_deg(self) -> float:
        return float(self.wind_direction_ann.value)

    @property
    def temperature_c(self) -> float:
        return float(self.temperature_ann.value)

    def snapshot(self) -> Dict[str, float]:
        return {
            "wind_speed": self.wind_speed,
            "wind_direction_deg": self.wind_direction_deg,
            "temperature_c": self.temperature_c,
            "live": self.wind_speed_ann.is_fresh(),
        }

    def __str__(self) -> str:
        s = self.snapshot()
        return (f"wind {s['wind_speed']:.1f} m/s FROM {s['wind_direction_deg']:.0f} deg, "
                f"{s['temperature_c']:.1f} degC"
                f"{'' if s['live'] else '  [not updating]'}")