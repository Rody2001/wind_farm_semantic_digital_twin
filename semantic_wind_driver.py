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
objects.

IMPORTANT, learned the hard way: in this fork, WorldState is keyed by
dof.id (a UUID) -- NOT dof.name (a PrefixedName). An earlier version of
this file used .name as the key, which always raised
DofNotInWorldStateError (a PrefixedName is never a valid key in a
UUID-indexed dict), and a "repair" that called
world.state.add_degree_of_freedom(dof) again made things worse: that
method also calls dof.create_variables(), replacing dof.variables.position
with a brand-new object -- but RevoluteConnection.add_to_world() had
already baked the *original* PositionVariable object into a cached
symbolic expression when the connection was first built. That mismatch
between the compiled expression and the compiled parameter list is what
produced HasFreeVariablesError. There was never anything actually missing
from world.state.

The fix: use dof.id as the key, and never call create_variables() again --
the DOF is already correctly registered the moment create_with_dofs() ran.
world.state[dof_id].position/.velocity are plain numpy writes (no
side effects), so we can batch-update every turbine and call
world.notify_state_change() once at the end of step() -- the semantic-world
equivalent of calling mj_forward() once at the end of QueryDriver.advance().
===================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection

import turbine_formulas as formulas
from wind_state_file import read_wind_state  # optional: follow wind from the MuJoCo viewer


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
                 rotor_accel_rpm_s: float = 1.0, follow_wind_file: str = None,
                 ros_subscriber=None):
        self.world = world
        self.turbines = turbines
        self.tsr = tsr
        self.yaw_rate_deg = yaw_rate_deg
        self.rotor_accel_rpm_s = rotor_accel_rpm_s
        self.wind_speed = 0.0
        self.wind_direction_deg = 0.0
        # If set, step() re-reads wind from this file every tick instead of only from
        # set_wind() -- lets a MuJoCo viewer (wind_turbine_sim.py) running in a separate
        # process drive this world's wind live. Manual set_wind() calls still work when
        # this is None (e.g. for standalone testing without MuJoCo running); when it's
        # set, the next step() overwrites them with whatever the file says, since the
        # file is treated as the source of truth. Pass None to disable and go back to
        # pure manual control.
        self.follow_wind_file = follow_wind_file
        # If set (a RosTurbineSubscriber), step() mirrors that turbine's published rpm
        # directly instead of recomputing it from wind physics here -- so this world's
        # numbers are exactly MuJoCo's, not an independent (and possibly diverging)
        # simulation. Falls back to our own wind-driven ramp until the first message
        # actually arrives for a turbine (e.g. before --publish is running), so queries
        # aren't stuck at a meaningless 0 in the meantime.
        self.ros_subscriber = ros_subscriber

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
        if self.follow_wind_file:
            self.wind_speed, self.wind_direction_deg = read_wind_state(self.follow_wind_file)
        speed, direction_deg = self.wind_speed, self.wind_direction_deg
        wind_vec = formulas.wind_from_bearing(speed, direction_deg)

        if speed > 1e-9:
            source_dir = -wind_vec[:2] / speed
            target_world_rad = float(np.arctan2(source_dir[1], source_dir[0]))
        else:
            target_world_rad = None    # no wind direction to chase; hold position

        max_yaw_step = np.deg2rad(self.yaw_rate_deg) * dt
        max_rpm_step = self.rotor_accel_rpm_s * dt

        # Write straight into world.state, keyed by dof.id (a UUID -- see module
        # docstring), so we only recompute forward kinematics ONCE per step for the
        # whole farm via a single notify_state_change() call at the end.
        for t in self.turbines.values():
            try:
                current_yaw = t.nacelle_conn.position
                if target_world_rad is not None:
                    target_local = _wrap_to_pi(target_world_rad - t.base_yaw_rad)
                    yaw_diff = _wrap_to_pi(target_local - current_yaw)
                    yaw_step = float(np.clip(yaw_diff, -max_yaw_step, max_yaw_step))
                else:
                    target_local = current_yaw
                    yaw_step = 0.0
                new_yaw = current_yaw + yaw_step
                nacelle_dof_id = t.nacelle_conn.raw_dof.id
                self.world.state[nacelle_dof_id].position = new_yaw
                self.world.state[nacelle_dof_id].velocity = (
                    yaw_step / dt if dt > 0 else 0.0)

                # Effective wind seen by the rotor: full speed once the nacelle is aligned,
                # smoothly less while it's still turning to catch up (angle_error -> 0 means
                # alignment -> 1). This is what makes RPM build up gradually after a wind
                # direction change, not just after a wind speed change.
                angle_error = (abs(_wrap_to_pi(target_local - new_yaw))
                               if target_world_rad is not None else np.pi)
                alignment = max(0.0, float(np.cos(angle_error)))
                effective_wind = speed * alignment

                published = self.ros_subscriber.get(t.name) if self.ros_subscriber else None
                if published is not None and published.rpm_seen:
                    # Ground truth from MuJoCo -- mirror it exactly, don't recompute.
                    t.current_rpm = published.rpm
                else:
                    # No ROS data yet (e.g. --publish isn't running): fall back to our
                    # own wind-driven ramp so queries aren't stuck at a meaningless 0.
                    target_rpm = formulas.rpm_for_wind(effective_wind, t.blade_length, self.tsr)
                    rpm_step = float(np.clip(target_rpm - t.current_rpm, -max_rpm_step, max_rpm_step))
                    t.current_rpm += rpm_step

                omega = t.current_rpm * 2.0 * np.pi / 60.0
                hub_dof_id = t.hub_conn.raw_dof.id
                self.world.state[hub_dof_id].position = self.world.state[hub_dof_id].position + omega * dt
                self.world.state[hub_dof_id].velocity = omega
            except Exception as exc:   # noqa: BLE001
                print(f"[SemanticWindDriver] skipping '{t.name}' this step: {exc!r}")

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
        """Snapshot of one turbine's (or every turbine's) live state -- the query.

        When a ros_subscriber is attached, rpm/power_w/energy_kwh here are exactly
        what wind_turbine_sim.py published for that turbine (see step() -- rpm is
        mirrored, not recomputed), plus the two farm-wide totals it also publishes.
        """
        speed, direction_deg = self.wind_speed, self.wind_direction_deg
        if name is not None:
            t = self.turbines[name]
            result = {
                "name": name,
                "rpm": t.current_rpm,
                "spinning": self.is_spinning(name),
                "nacelle_yaw_deg": self.nacelle_yaw_deg(name),
                "wind_speed": speed,
                "wind_direction_deg": direction_deg,
            }
            if self.ros_subscriber is not None:
                published = self.ros_subscriber.get(name)
                result["power_w"] = published.power_w
                result["energy_kwh"] = published.energy_kwh
            return result
        result = {
            "wind_speed": speed,
            "wind_direction_deg": direction_deg,
            "turbines": {n: self.status(n) for n in self.turbines},
        }
        if self.ros_subscriber is not None:
            result["total_power_w"] = self.ros_subscriber.total_power_w
            result["total_energy_kwh"] = self.ros_subscriber.total_energy_kwh
        return result


# --------------------------------------------------------------------------- #
# module-level convenience wrappers (so callers can write set_wind(driver, ...))
# --------------------------------------------------------------------------- #
def set_wind(driver: SemanticWindDriver, speed: float, direction_deg: float) -> None:
    driver.set_wind(speed, direction_deg)


def get_wind(driver: SemanticWindDriver) -> Tuple[float, float]:
    return driver.get_wind()