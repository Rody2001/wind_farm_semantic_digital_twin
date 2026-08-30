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
===================================================================
"""

from __future__ import annotations

import copy
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from peak_state_file import read_peak_state, write_peak_state

import turbine_formulas as formulas
from wind_state_file import read_wind_state  # optional: follow wind from the MuJoCo viewer


# --------------------------------------------------------------------------- #
# per-turbine runtime record
# --------------------------------------------------------------------------- #
@dataclass
class TurbineRuntime:
    """Everything the driver needs to animate + query one turbine."""
    name: str
    hub_conn: RevoluteConnection
    nacelle_conn: RevoluteConnection
    blade_length: float
    base_yaw_rad: float
    # Rating: at and above max_kw_wind_speed [m/s] this turbine holds max_kw [kW]
    # instead of following the cp curve; at and above cut_out_speed it produces
    # nothing. All three are derived from blade_length when left at 0 -- see
    # __post_init__ and turbine_formulas.rating_for_length().
    max_kw: float = 0.0
    max_kw_wind_speed: float = 0.0
    cut_out_speed: float = 0.0
    current_rpm: float = 0.0
    current_power: float = 0.0
    current_energy_kwh: float = 0.0
    # measured annotations in the World, filled in by annotate_turbines()
    rotor_speed_ann: object = None
    nacelle_yaw_ann: object = None
    power_ann: object = None
    energy_ann: object = None

    def __post_init__(self):
        """Fill in whatever rating the caller left at 0, from the blade length.

        Doing it once here means every query (status(), at_rated_power, ...) sees
        the real numbers instead of a placeholder 0, whoever built the runtime.
        """
        self.max_kw, self.max_kw_wind_speed = formulas.rating_for_length(
            self.blade_length, self.max_kw, self.max_kw_wind_speed)
        if not self.cut_out_speed:
            self.cut_out_speed = formulas.cut_out_for_length(self.blade_length)


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
                 ros_subscriber=None, peak_file: str = "peak_state.json",
                 environment=None):
        self.world = world
        self.turbines = turbines
        self.tsr = tsr
        self.yaw_rate_deg = yaw_rate_deg
        self.rotor_accel_rpm_s = rotor_accel_rpm_s
        self.wind_speed = 0.0
        self.wind_direction_deg = 0.0
        self.follow_wind_file = follow_wind_file
        self.ros_subscriber = ros_subscriber
        # The three environment annotations in the World. step() writes the current
        # wind into them, the same way it writes turbine poses into world.state.
        self.environment = environment

        # ---- peak tracking (persisted across process restarts) ----
        self.elapsed = 0.0                  # sim time accumulated across step() calls
        self.peak_file = peak_file
        stored = read_peak_state(peak_file) if peak_file else None
        self.peak_status = stored
        self.peak_power_w = stored["total_power_w"] if stored else 0.0


    def set_wind(self, speed: float, direction_deg: float) -> None:
        """Set the live wind speed [m/s] and direction [deg, compass bearing wind is FROM].

        Call this any time -- from a REPL, a query script, whatever -- and the next
        step() will start steering every turbine toward the new conditions.
        """
        self.wind_speed = speed
        self.wind_direction_deg = direction_deg % 360.0


    def step(self, dt: float) -> None:
        """Advance every turbine by dt seconds using the driver's current wind state."""
        if self.follow_wind_file:
            self.wind_speed, self.wind_direction_deg = read_wind_state(self.follow_wind_file)
        speed, direction_deg = self.wind_speed, self.wind_direction_deg
        if self.environment is not None:
            self.environment.update(wind_speed=speed, wind_direction_deg=direction_deg)
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
                    t.current_power = published.power_w
                    t.current_energy_kwh = published.energy_kwh
                else:
                    # No ROS data yet (e.g. --publish isn't running): fall back to our
                    # own wind-driven ramp so queries aren't stuck at a meaningless 0.
                    target_rpm = formulas.rpm_for_wind(effective_wind, t.blade_length, self.tsr)
                    rpm_step = float(np.clip(target_rpm - t.current_rpm, -max_rpm_step, max_rpm_step))
                    t.current_rpm += rpm_step
                    # capped_power_for_length holds this turbine's rated output above
                    # its rated wind speed, and returns 0 above its own cut-out
                    # speed -- both scaled to this turbine's blade length.
                    t.current_power = (0.0 if abs(t.current_rpm) < 1e-9 else
                                       formulas.capped_power_for_length(
                                           formulas.RHO, effective_wind, t.blade_length,
                                           t.max_kw, t.max_kw_wind_speed))
                # The same values, now as properties of the bodies they belong to,
                # which is where every query reads them from.
                if t.rotor_speed_ann is not None:
                    t.rotor_speed_ann.set(t.current_rpm)
                    t.power_ann.set(t.current_power)
                    t.energy_ann.set(t.current_energy_kwh)
                    t.nacelle_yaw_ann.set(np.degrees(new_yaw))

                omega = t.current_rpm * 2.0 * np.pi / 60.0
                hub_dof_id = t.hub_conn.raw_dof.id
                self.world.state[hub_dof_id].position = self.world.state[hub_dof_id].position + omega * dt
                self.world.state[hub_dof_id].velocity = omega
            except Exception as exc:   # noqa: BLE001
                print(f"[SemanticWindDriver] skipping '{t.name}' this step: {exc!r}")

        self.world.notify_state_change()   # one recompute for the whole farm, like mj_forward()

        self.elapsed += dt
        total = self.total_power_w()

        # ---- peak: only when a NEW maximum is reached ----
        if total > self.peak_power_w:
            self.peak_power_w = total
            snapshot = copy.deepcopy(self.status())
            snapshot["time_s"] = self.elapsed
            snapshot["timestamp"] = datetime.now().isoformat(timespec="seconds")
            snapshot["total_power_w"] = total
            self.peak_status = snapshot
            if self.peak_file:
                write_peak_state(snapshot, self.peak_file)

    # ---- queries ---------------------------------------------------------- #
    def is_spinning(self, name: str, eps: float = 0.05) -> bool:
        """True if turbine `name`'s rotor RPM is above a negligible threshold."""
        return abs(self.turbines[name].current_rpm) > eps

    def total_power_w(self) -> float:
        """Total generated power [W] across the whole farm right now."""
        return sum(t.current_power for t in self.turbines.values())

    def rpm(self, name: str) -> float:
        """Current (ramped) RPM of turbine `name`."""
        return self.turbines[name].current_rpm

    def nacelle_yaw_deg(self, name: str) -> float:
        """Current nacelle yaw angle (local, relative to the tower) in degrees."""
        return float(np.degrees(self.turbines[name].nacelle_conn.position))

    def reset_peak(self) -> None:
        """Forget the recorded peak (e.g. when starting a new experiment)."""
        self.peak_power_w = 0.0
        self.peak_status = None

    def status(self, name: str = None) -> Dict:
        """Snapshot of one turbine's (or every turbine's) live state -- the query.

        rpm/power_w/energy_kwh come from TurbineRuntime, which step() fills either
        from the published ROS values (ground truth from wind_turbine_sim.py) or
        from the local wind-driven fallback. Reading the stored fields instead of
        the subscriber avoids an AttributeError for turbines that have not
        published yet (subscriber.get(name) returns None in that case).
        """
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
                "power_w": t.current_power,
                "energy_kwh": t.current_energy_kwh,
                "max_kw": t.max_kw,
                "max_kw_wind_speed": t.max_kw_wind_speed,
                "cut_out_speed": t.cut_out_speed,
                # True while this turbine is sitting on its rating rather than
                # following the cp curve.
                "at_rated_power": bool(t.max_kw_wind_speed <= speed < t.cut_out_speed),
            }
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