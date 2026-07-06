"""
semantic_wind_driver.py
===================================================================
Drives the semantic-digital-twin World the same way QueryDriver drives
the MuJoCo world (wind_turbine_sim.py): reads a live wind speed +
direction, yaws each nacelle to face it, ramps rotor RPM up/down with
inertia instead of snapping, and lets callers query "is this turbine
spinning" / "what's its RPM right now" at any point in time.

Both drivers share the exact same physics from turbine_formulas.py
(rpm_for_wind, wind_from_bearing, ...), so the semantic world and the
MuJoCo world behave identically for the same wind input.

No MuJoCo import here -- this module only touches semantic_digital_twin
objects, verified directly against the installed package's real API:
  - DegreeOfFreedom state lives in world.state[dof.name], NOT dof.id.
  - RevoluteConnection.position / .velocity read+write that state and
    call world.notify_state_change() for you (like mj_forward()).
  - world.notify_state_change() re-triggers forward kinematics, which is
    what the existing VizMarkerPublisher/tf publisher reads from -- so a
    single call per step() is what actually makes the model visibly move
    in RViz.
  - IMPORTANT: the World rejects DegreeOfFreedom instances that aren't
    used by any Connection (validated when a modify_world() block exits).
    So wind speed/direction are NOT world DOFs here -- they're plain
    attributes on the driver itself, set via set_wind()/get_wind(). This
    is exactly the trap the old "commented out, because the world doesn't
    like DoFs that are not linked to a connection" note in main1.py was
    warning about.
===================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection

import turbine_formulas as formulas


# --------------------------------------------------------------------------- #
# per-turbine runtime record
# --------------------------------------------------------------------------- #
@dataclass
class TurbineRuntime:
    """Everything the driver needs to animate + query one turbine."""
    name: str
    hub_conn: RevoluteConnection        # rotor spin (axis X, child of nacelle)
    nacelle_conn: RevoluteConnection    # yaw (axis Z, child of tower)
    blade_length: float
    base_yaw_rad: float                 # fixed world yaw of the tower (farm layout orientation)
    current_rpm: float = 0.0


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi], for shortest-path yaw slewing."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
class SemanticWindDriver:
    """Live wind -> nacelle yaw + rotor RPM driver for the semantic-digital-twin World.

    Call step(dt) repeatedly (e.g. from a ROS2 timer) to animate every turbine.
    Call the query methods any time in between to read the current state.
    Wind is plain state on this driver (see set_wind/get_wind) -- NOT a World DOF,
    since the framework requires every DOF to belong to a Connection.
    """

    def __init__(self, world: World, turbines: Dict[str, TurbineRuntime],
                 tsr: float = formulas.TSR, yaw_rate_deg: float = 10.0,
                 rotor_accel_rpm_s: float = 1.0):
        self.world = world
        self.turbines = turbines
        self.tsr = tsr
        self.yaw_rate_deg = yaw_rate_deg
        self.rotor_accel_rpm_s = rotor_accel_rpm_s
        self.wind_speed = 0.0
        self.wind_direction_deg = 0.0

    def set_wind(self, speed: float, direction_deg: float) -> None:
        """Set the live wind speed [m/s] and direction [deg, compass bearing wind is FROM].

        Call this any time -- from a REPL, a query script, whatever -- and the next
        step() will start steering every turbine toward the new conditions.
        """
        self.wind_speed = speed
        self.wind_direction_deg = direction_deg % 360.0

    def get_wind(self) -> Tuple[float, float]:
        """Return (speed [m/s], direction [deg]) currently set on this driver."""
        return self.wind_speed, self.wind_direction_deg

    def step(self, dt: float) -> None:
        """Advance every turbine by dt seconds using the driver's current wind state."""
        speed, direction_deg = self.wind_speed, self.wind_direction_deg
        wind_vec = formulas.wind_from_bearing(speed, direction_deg)

        if speed > 1e-9:
            source_dir = -wind_vec[:2] / speed
            target_world_rad = float(np.arctan2(source_dir[1], source_dir[0]))
        else:
            target_world_rad = None    # no wind direction to chase; hold position

        max_yaw_step = np.deg2rad(self.yaw_rate_deg) * dt
        max_rpm_step = self.rotor_accel_rpm_s * dt

        # Write straight into world.state (bypassing the per-connection setters, which
        # each call notify_state_change()) so we only recompute forward kinematics ONCE
        # per step for the whole farm -- the semantic-world equivalent of calling
        # mj_forward() once at the end of QueryDriver.advance().
        for t in self.turbines.values():
            current_yaw = t.nacelle_conn.position
            if target_world_rad is not None:
                target_local = _wrap_to_pi(target_world_rad - t.base_yaw_rad)
                yaw_diff = _wrap_to_pi(target_local - current_yaw)
                yaw_step = float(np.clip(yaw_diff, -max_yaw_step, max_yaw_step))
            else:
                target_local = current_yaw
                yaw_step = 0.0
            new_yaw = current_yaw + yaw_step
            self.world.state[t.nacelle_conn.raw_dof.name].position = new_yaw
            self.world.state[t.nacelle_conn.raw_dof.name].velocity = (
                yaw_step / dt if dt > 0 else 0.0)

            # Effective wind seen by the rotor: full speed once the nacelle is aligned,
            # smoothly less while it's still turning to catch up (angle_error -> 0 means
            # alignment -> 1). This is what makes RPM build up gradually after a wind
            # direction change, not just after a wind speed change.
            angle_error = abs(_wrap_to_pi(target_local - new_yaw)) if target_world_rad is not None else np.pi
            alignment = max(0.0, float(np.cos(angle_error)))
            effective_wind = speed * alignment

            target_rpm = formulas.rpm_for_wind(effective_wind, t.blade_length, self.tsr)
            rpm_step = float(np.clip(target_rpm - t.current_rpm, -max_rpm_step, max_rpm_step))
            t.current_rpm += rpm_step

            omega = t.current_rpm * 2.0 * np.pi / 60.0
            hub_name = t.hub_conn.raw_dof.name
            self.world.state[hub_name].position = self.world.state[hub_name].position + omega * dt
            self.world.state[hub_name].velocity = omega

        self.world.notify_state_change()   # one recompute for the whole farm, like mj_forward()

    # ---- queries ---------------------------------------------------------- #
    def is_spinning(self, name: str, eps: float = 0.05) -> bool:
        """True if turbine `name`'s rotor RPM is above a negligible threshold."""
        return abs(self.turbines[name].current_rpm) > eps

    def rpm(self, name: str) -> float:
        """Current (ramped) RPM of turbine `name`."""
        return self.turbines[name].current_rpm

    def nacelle_yaw_deg(self, name: str) -> float:
        """Current nacelle yaw angle (local, relative to the tower) in degrees."""
        return float(np.degrees(self.turbines[name].nacelle_conn.position))

    def status(self, name: str = None) -> Dict:
        """Snapshot of one turbine's (or every turbine's) live state -- the query."""
        speed, direction_deg = self.wind_speed, self.wind_direction_deg
        if name is not None:
            t = self.turbines[name]
            return {
                "name": name,
                "rpm": t.current_rpm,
                "spinning": self.is_spinning(name),
                "nacelle_yaw_deg": self.nacelle_yaw_deg(name),
                "wind_speed": speed,
                "wind_direction_deg": direction_deg,
            }
        return {
            "wind_speed": speed,
            "wind_direction_deg": direction_deg,
            "turbines": {n: self.status(n) for n in self.turbines},
        }


# --------------------------------------------------------------------------- #
# module-level convenience wrappers (so callers can write set_wind(driver, ...))
# --------------------------------------------------------------------------- #
def set_wind(driver: SemanticWindDriver, speed: float, direction_deg: float) -> None:
    driver.set_wind(speed, direction_deg)


def get_wind(driver: SemanticWindDriver) -> Tuple[float, float]:
    return driver.get_wind()

