"""
wind_turbine_sim.py
===================================================================
Drive the MuJoCo wind turbines (query-driven RPM by default).

Adds optional ROS 2 publishing of live RPM + generated power/energy:

    python wind_turbine_sim.py --publish            # publish to ROS 2 topics
    python wind_turbine_sim.py --needed 5 --publish  # demand cap + publish

Topics (std_msgs/Float64), one per turbine plus farm totals:
    /wind_farm/<turbine>/rpm
    /wind_farm/<turbine>/power_w      generated power now (0 if idle)
    /wind_farm/<turbine>/energy_kwh   cumulative since start
    /wind_farm/<turbine>/cp           power coefficient cp(v) for that turbine's current wind (0 if idle)
    /wind_farm/total_power_w
    /wind_farm/total_energy_kwh
    /wind_farm/temperature_c          current air temperature (deg C)
    /wind_farm/rho                    current air density (kg/m^3), from temperature_c
    /wind_farm/wind_speed             current wind speed (m/s)
    /wind_farm/wind_direction_deg     bearing the wind comes FROM (0=N, 90=E, 180=S, 270=W)
    /wind_farm/grid_limit_mw          current grid upper limit (MW)

Browser control panel:
    --ui               serve control_panel.html and open it in a browser; the sliders and
                       the compass change wind speed, wind direction, temperature and the
                       grid limit while the simulation runs
    --ui-port 8080     port for that panel (default 8080)
The viewer keys below keep working exactly as before: both the keys and the browser
write into one shared EnvironmentState, so a change made on either side shows up on
the other. --ui also works together with --headless, i.e. control without a viewer.

Timed run + history logging:
    --time 100         run 100 s then close automatically, logging history at 1 Hz
    --history-file PATH where to write the 1 Hz log (default history.jsonl); the queries
                        in history_queries.py read it back (spinning_intervals, ...)

Other options:
    --wind 8 --tsr 6 --axial --aero --headless N
    --cp 0.45         optional: override the polynomial cp(v) curve with a fixed value
    --yaw-rate 10     nacelle yaw slew speed in deg/s (default 10) while it turns to face the wind
    --rotor-accel 1.0 max RPM/s the blades can speed up or slow down (default 1.0) -- rotor
                      inertia, so wind gusts/lulls cause a gradual ramp, not an instant jump
    --temp 15         initial air temperature in deg C (default 15 -> rho=1.225 kg/m^3),
                      sets air density (rho) via the ideal gas law
Viewer keys:
    S = wind-SPEED mode, then Up/Down = +/- 1 m/s
    D = wind-DIRECTION mode, then Up/Down = +/- 5 deg (0-360, wraps around)
    T = TEMPERATURE mode, then Up/Down = +/- 1 degC -- recomputes air density (rho) live
    G = GRID-LIMIT mode, then Up/Down = +/- 50 MW
    R = reset (also zeros energy)
Each turbine's nacelle automatically yaws to face wherever the wind is currently
coming from, slewing smoothly rather than snapping to the new heading. Rotor RPM
and generated power likewise ramp up/down toward their wind-driven target instead
of snapping there instantly.
===================================================================
"""

import argparse
import os
import time
from datetime import datetime
import numpy as np
import mujoco

import turbine_formulas as formulas  # pure cut-in + RPM + power formulas (no framework)
from peak_state_file import clear_peak_state
from wind_state_file import write_wind_state  # shares live wind with main1.py's semantic world

try:
    from history_file import append_sample, clear_history   # optional 1 Hz history logging
    _HISTORY_OK = True
except Exception:  # noqa: BLE001
    _HISTORY_OK = False

from control_server import EnvironmentState, start_control_server, set_field_range

RHO = formulas.RHO   # air density [kg/m^3]
CD  = 1.28           # flat-plate drag coeff  (aero mode only)


def _wrap_to_pi(angle):
    """Wrap an angle in radians to (-pi, pi], for shortest-path yaw slewing."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =================================================================== #
# QUERY-DRIVEN KINEMATIC MODEL (default)
# =================================================================== #
class QueryDriver:
    """RPM from formulas.rpm_for_wind, gated by cut-in and an optional demand cap.
    Tracks generated power and accumulated energy for publishing."""

    def __init__(self, model: mujoco.MjModel, axial: bool = False, tsr: float = formulas.TSR,
                 grid_limit_mw: float = None, rho: float = formulas.RHO, c_p: float = None,
                 facing: bool = False, yaw_rate_deg: float = 10.0, rotor_accel_rpm_s: float = 1.0):
        self.model = model
        self.axial = axial
        self.facing = facing        # if True: only turbines facing the wind spin
        self.facing_cos = 0.5       # alignment threshold (cos 60 deg)
        self.tsr = tsr
        self.rho = rho
        self.c_p = c_p
        self.yaw_rate_deg = yaw_rate_deg   # nacelle slew speed while tracking the wind
        self.rotor_accel_rpm_s = rotor_accel_rpm_s   # max RPM change per second (rotor inertia)
        # grid upper limit [W]; None -> no cap, every eligible turbine runs
        self.grid_limit_w = None if grid_limit_mw is None else grid_limit_mw * 1e6
        self.rotors = []            # (name, body_id, qpos_adr, dof_adr, axis_local, D)
        self.yaws = []               # (name, qpos_adr, dof_adr, base_yaw_rad)

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            if name.endswith("_rotor"):
                body_id = model.jnt_bodyid[j]
                qpadr   = model.jnt_qposadr[j]
                dadr    = model.jnt_dofadr[j]
                axis    = model.jnt_axis[j].copy()
                D       = self._blade_length(body_id)
                self.rotors.append((name, body_id, qpadr, dadr, axis, D))
            elif name.endswith("_yaw"):
                nacelle_body_id = model.jnt_bodyid[j]
                qpadr = model.jnt_qposadr[j]
                dadr  = model.jnt_dofadr[j]
                base_body_id = model.body_parentid[nacelle_body_id]
                # base body is welded with a pure Z-axis euler rotation, so its quaternion
                # (w, x, y, z) has only w and z populated: yaw = 2 * atan2(z, w).
                bq = model.body_quat[base_body_id]
                base_yaw = 2.0 * float(np.arctan2(bq[3], bq[0]))
                self.yaws.append((name, qpadr, dadr, base_yaw))

        if not self.rotors:
            raise RuntimeError("No '*_rotor' joints found in the model.")

        # rotor joint name -> (max_kw [kW], max_kw_wind_speed [m/s]); see below
        self.ratings = self._read_ratings()

        self.names = [r[0] for r in self.rotors]
        self.energy_wh = {n: 0.0 for n in self.names}    # cumulative per turbine
        self.total_energy_wh = 0.0
        self.rotor_rpm = {k: 0.0 for k in range(len(self.rotors))}   # actual (ramped) RPM per rotor
        self.last_readings = {}                          # name -> (rpm, power_w, energy_wh, active, cp)
        self.last_total_power = 0.0
        # power the wind could give right now, before the grid cap is applied. The
        # difference to last_total_power is what curtailment throws away.
        self.last_available_power = 0.0

    def _blade_length(self, hub_body_id):
        for g in range(self.model.ngeom):
            gb = self.model.geom_bodyid[g]
            if self.model.body_parentid[gb] != hub_body_id:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.endswith("_geom"):
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    # For a mesh geom, geom_size is the half-extent measured from
                    # the recentred mesh origin, which is not the span. geom_aabb
                    # is the real bounding box, and a blade is far longer than it
                    # is wide, so its largest extent is the length.
                    return 2.0 * float(max(self.model.geom_aabb[g][3:6]))
                return 2.0 * float(self.model.geom_size[g][2])
        return 1.0

    def _read_ratings(self):
        """Per-turbine (max_kw, max_kw_wind_speed), read from the scene.

        wind_farm_export.py writes one <custom><numeric name="<turbine>_rating"
        data="max_kw max_kw_wind_speed"/> per rated turbine, so the rating
        travels with the MJCF instead of having to be looked up in the Python
        specs -- this driver then caps exactly the same turbines at exactly the
        same speeds as SemanticWindDriver does. Turbines with no entry (older
        scenes included) stay uncapped and keep following the cp curve.
        """
        table = {}
        for i in range(self.model.nnumeric):
            key = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
            if not key or not key.endswith("_rating"):
                continue
            adr = self.model.numeric_adr[i]
            size = self.model.numeric_size[i]
            data = [float(x) for x in self.model.numeric_data[adr:adr + size]]
            max_kw = data[0] if len(data) > 0 else 0.0
            rated_v = data[1] if len(data) > 1 else 0.0
            table[f"{key[:-len('_rating')]}_rotor"] = (max_kw, rated_v)
        return table

    def rating(self, rotor_joint_name):
        """(max_kw, max_kw_wind_speed) for a rotor joint; (0, 0) if unrated."""
        return self.ratings.get(rotor_joint_name, (0.0, 0.0))

    def reset_energy(self):
        for n in self.names:
            self.energy_wh[n] = 0.0
        self.total_energy_wh = 0.0

    def _update_yaw(self, data, wind_vec, dt):
        """Slew each nacelle's yaw joint toward facing the current wind direction.

        Moves at most yaw_rate_deg * dt degrees per step, so the nacelle visibly
        turns to the correct heading instead of snapping there instantly.
        """
        if not self.yaws:
            return
        speed = float(np.linalg.norm(wind_vec[:2]))
        if speed < 1e-9:
            return                              # no wind direction to chase, hold position
        source_dir = -wind_vec[:2] / speed      # unit vector toward where the wind comes from
        target_world = float(np.arctan2(source_dir[1], source_dir[0]))
        max_step = np.deg2rad(self.yaw_rate_deg) * dt
        for _name, qpadr, dadr, base_yaw in self.yaws:
            target_local = _wrap_to_pi(target_world - base_yaw)
            current = float(data.qpos[qpadr])
            diff = _wrap_to_pi(target_local - current)
            step = float(np.clip(diff, -max_step, max_step))
            data.qpos[qpadr] = current + step
            data.qvel[dadr] = step / dt if dt > 0 else 0.0

    def _rotor_wind(self, data, body_id, axis_local, wind_vec):
        if self.facing:
            # only spin if the rotor axis points toward where the wind comes from
            speed = float(np.linalg.norm(wind_vec))
            if speed < 1e-9:
                return 0.0
            R = data.xmat[body_id].reshape(3, 3)
            axis_world = R @ axis_local
            wind_from = -wind_vec / speed          # unit vector toward the source
            align = float(np.dot(axis_world, wind_from))
            return speed if align > self.facing_cos else 0.0
        if self.axial:
            R = data.xmat[body_id].reshape(3, 3)
            return float(np.dot(wind_vec, R @ axis_local))
        return float(np.linalg.norm(wind_vec))

    def _active_set(self, data, wind_vec):
        rows = []
        for k, (name, body_id, _, _, axis_local, D) in enumerate(self.rotors):
            v = self._rotor_wind(data, body_id, axis_local, wind_vec)
            rpm = formulas.rpm_for_wind(v, D, self.tsr)
            cp = formulas.cp_for_wind(v) if self.c_p is None else self.c_p
            max_kw, rated_v = self.rating(name)
            # capped_power_for_length holds the rated output above rated_v and
            # returns 0 above the 28 m/s cut-out; unrated turbines fall through
            # to the plain cp curve, exactly as before.
            power = (formulas.capped_power_for_length(self.rho, v, D, max_kw, rated_v, self.c_p)
                     if rpm != 0 else 0.0)
            rows.append((k, rpm, power, cp))

        if self.grid_limit_w is None:
            return {k for k, rpm, _, _ in rows if rpm != 0}, rows

        # Grid cap: run as many turbines as fit WITHOUT the total generated
        # power exceeding the limit (largest first, skip any that would push over).
        active, cum = set(), 0.0
        for k, rpm, power, cp in sorted(rows, key=lambda r: (-r[2], self.rotors[r[0]][0])):
            if power <= 0:
                continue
            if cum + power <= self.grid_limit_w:
                active.add(k)
                cum += power
            # else skip: adding this turbine would exceed the grid limit
        return active, rows

    def advance(self, data: mujoco.MjData, wind_vec: np.ndarray, dt: float):
        self._update_yaw(data, wind_vec, dt)
        active, rows = self._active_set(data, wind_vec)
        self.last_available_power = sum(r[2] for r in rows)
        total_power = 0.0
        max_rpm_step = self.rotor_accel_rpm_s * dt
        for k, target_rpm, target_power, target_cp in rows:
            name, body_id, qpadr, dadr, axis_local, D = self.rotors[k]
            is_on = k in active
            target_rpm_eff = target_rpm if is_on else 0.0   # idle/off rotors coast down to 0

            current_rpm = self.rotor_rpm[k]
            step = float(np.clip(target_rpm_eff - current_rpm, -max_rpm_step, max_rpm_step))
            current_rpm += step
            self.rotor_rpm[k] = current_rpm

            omega = current_rpm * 2.0 * np.pi / 60.0
            data.qvel[dadr] = omega
            data.qpos[qpadr] += omega * dt

            # power builds up with RPM instead of jumping straight to the target
            ramp_frac = 0.0 if target_rpm_eff <= 1e-9 else min(1.0, current_rpm / target_rpm_eff)
            gen = target_power * ramp_frac if is_on else 0.0
            cp_now = target_cp if is_on else 0.0
            self.energy_wh[name] += gen * dt / 3600.0
            self.last_readings[name] = (current_rpm, gen, self.energy_wh[name], is_on, cp_now)
            total_power += gen
        self.total_energy_wh += total_power * dt / 3600.0
        self.last_total_power = total_power
        mujoco.mj_forward(self.model, data)
        data.time += dt
        return total_power

    def rpm(self, data):
        return {name: float(data.qvel[dadr]) * 60.0 / (2 * np.pi)
                for name, _, _, dadr, _, _ in self.rotors}

    def info(self):
        return [(name, D, formulas.min_wind_speed_for_length(D, self.tsr))
                for name, _, _, _, _, D in self.rotors]

    def preview(self, data, wind_vec):
        active, rows = self._active_set(data, wind_vec)
        used = sum(p for k, _, p, _ in rows if k in active)
        total = sum(p for _, _, p, _ in rows)
        names_on = [self.rotors[k][0] for k, _, _, _ in rows if k in active]
        return active, used, total, names_on


# =================================================================== #
# ORIGINAL AERODYNAMIC FORCE MODEL  (--aero)
# =================================================================== #
class Aerodynamics:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.rho = RHO   # mutable so temperature control ('T' key) can update it live
        self.grid_limit_w = None   # no curtailment in the force model, but the attribute
                                   # has to exist so the shared controls can address it
        self.blades = []
        for s in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)
            if name is None or not name.endswith("_aero"):
                continue
            body_id = model.site_bodyid[s]
            self.blades.append((s, body_id, self._plate_area(body_id)))
        if not self.blades:
            raise RuntimeError("No '*_aero' sites found in the model.")
        self.rotors = []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.endswith("_rotor"):
                self.rotors.append((name, model.jnt_dofadr[j]))

    def _plate_area(self, body_id):
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != body_id:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.endswith("_geom"):
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    # the two largest extents of the bounding box: chord x span
                    h = sorted(self.model.geom_aabb[g][3:6])
                    return (2 * h[1]) * (2 * h[2])
                sx, sy, sz = self.model.geom_size[g]
                return (2 * sy) * (2 * sz)
        return 0.1

    def apply(self, data, wind_vec):
        data.xfrc_applied[:] = 0.0
        vel6 = np.zeros(6)
        for site_id, body_id, area in self.blades:
            p = data.site_xpos[site_id]
            n = data.site_xmat[site_id].reshape(3, 3)[:, 0]
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel6, 0)
            v_rel = wind_vec - vel6[3:6]
            v_n = float(np.dot(v_rel, n))
            F = 0.5 * self.rho * CD * area * v_n * abs(v_n) * n
            r = p - data.xipos[body_id]
            data.xfrc_applied[body_id, 0:3] += F
            data.xfrc_applied[body_id, 3:6] += np.cross(r, F)

    def advance(self, data, wind_vec, dt):
        self.apply(data, wind_vec)
        mujoco.mj_step(self.model, data)

    def rpm(self, data):
        return {name: float(data.qvel[adr]) * 60.0 / (2 * np.pi)
                for name, adr in self.rotors}


# =================================================================== #
# ROS 2 PUBLISHER (optional)
# =================================================================== #
class RosPublisher:
    """Publishes per-turbine rpm/power/energy + farm totals on std_msgs/Float64."""

    def __init__(self, rotor_names):
        import rclpy                       # imported lazily so ROS is optional
        from std_msgs.msg import Float64
        self._rclpy = rclpy
        self._Float64 = Float64
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("wind_turbine_sim")
        self.rpm_pubs, self.power_pubs, self.energy_pubs, self.cp_pubs = {}, {}, {}, {}
        for n in rotor_names:
            base = n[:-6] if n.endswith("_rotor") else n     # strip "_rotor"
            self.rpm_pubs[n]    = self.node.create_publisher(Float64, f"/wind_farm/{base}/rpm", 10)
            self.power_pubs[n]  = self.node.create_publisher(Float64, f"/wind_farm/{base}/power_w", 10)
            self.energy_pubs[n] = self.node.create_publisher(Float64, f"/wind_farm/{base}/energy_kwh", 10)
            self.cp_pubs[n]     = self.node.create_publisher(Float64, f"/wind_farm/{base}/cp", 10)
        self.total_power_pub  = self.node.create_publisher(Float64, "/wind_farm/total_power_w", 10)
        self.total_energy_pub = self.node.create_publisher(Float64, "/wind_farm/total_energy_kwh", 10)
        self.temperature_pub  = self.node.create_publisher(Float64, "/wind_farm/temperature_c", 10)
        self.rho_pub          = self.node.create_publisher(Float64, "/wind_farm/rho", 10)
        self.wind_speed_pub   = self.node.create_publisher(Float64, "/wind_farm/wind_speed", 10)
        self.wind_dir_pub     = self.node.create_publisher(Float64, "/wind_farm/wind_direction_deg", 10)
        self.grid_limit_pub   = self.node.create_publisher(Float64, "/wind_farm/grid_limit_mw", 10)

    def publish(self, driver):
        F = self._Float64
        for name, (rpm, power, energy_wh, _active, cp) in driver.last_readings.items():
            self.rpm_pubs[name].publish(F(data=float(rpm)))
            self.power_pubs[name].publish(F(data=float(power)))
            self.energy_pubs[name].publish(F(data=float(energy_wh / 1000.0)))
            self.cp_pubs[name].publish(F(data=float(cp)))
        self.total_power_pub.publish(F(data=float(driver.last_total_power)))
        self.total_energy_pub.publish(F(data=float(driver.total_energy_wh / 1000.0)))
        temp_c = getattr(driver, "temp_c", None)
        if temp_c is not None:
            self.temperature_pub.publish(F(data=float(temp_c)))
        rho = getattr(driver, "rho", None)
        if rho is not None:
            self.rho_pub.publish(F(data=float(rho)))
        wind_speed = getattr(driver, "wind_speed", None)
        if wind_speed is not None:
            self.wind_speed_pub.publish(F(data=float(wind_speed)))
        wind_direction = getattr(driver, "wind_direction", None)
        if wind_direction is not None:
            self.wind_dir_pub.publish(F(data=float(wind_direction)))
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        grid_limit_w = getattr(driver, "grid_limit_w", None)
        if grid_limit_w is not None:
            self.grid_limit_pub.publish(F(data=float(grid_limit_w / 1e6)))

    def shutdown(self):
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def make_wind(speed, heading_deg=0.0):
    a = np.deg2rad(heading_deg)
    return np.array([np.cos(a), np.sin(a), 0.0]) * speed


# wind_from_bearing now lives in turbine_formulas.py (shared with the semantic-digital-twin
# driver so both worlds agree on the same wind-direction convention).
wind_from_bearing = formulas.wind_from_bearing


def _write_history(driver, path, elapsed, wind_speed, direction_deg):
    """Append one history sample in the schema history_queries.py expects.

    Turbine names are stripped of the trailing '_rotor' so the file uses the
    same names as the semantic side ('Farm_East_tall', not '..._rotor').
    """
    if not path or not hasattr(driver, "last_readings"):
        return
    def base(n):
        return n[:-6] if n.endswith("_rotor") else n
    limit_w = getattr(driver, "grid_limit_w", None)
    sample = {
        "time_s": float(elapsed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wind_speed": float(wind_speed),
        "wind_direction_deg": float(direction_deg),
        "total_power_w": float(getattr(driver, "last_total_power", 0.0)),
        "total_energy_kwh": float(getattr(driver, "total_energy_wh", 0.0)) / 1000.0,
        "grid_limit_mw": float(limit_w / 1e6) if limit_w is not None else None,
        "rpm": {base(n): r[0] for n, r in driver.last_readings.items()},
        "power_w": {base(n): r[1] for n, r in driver.last_readings.items()},
        "cp": {base(n): r[4] for n, r in driver.last_readings.items()},
    }
    try:
        append_sample(sample, path)
    except Exception as exc:   # noqa: BLE001
        print(f"[history] write failed: {exc!r}")


# =================================================================== #
# SHARED ENVIRONMENT  (viewer keys + browser panel write into the same state)
# =================================================================== #
SPEED_STEP = 0.5    # m/s per arrow press
DIR_STEP = 5.0       # degrees per arrow press
TEMP_STEP = 1.0      # deg C per arrow press
GRIDLIMIT_STEP = 50.0  # MW per arrow press
DEFAULT_GRIDLIMIT_MW = 500.0  # default grid limit in MW
DEFAULT_TEMP_C = 15.0   # matches RHO = 1.225 kg/m^3 (ENERCON's "Standardluftdichte")


def ui_requested(args):
    """True for --ui, or for WIND_TURBINE_UI=1 in the environment.

    The environment variable exists because the simulation is not always started
    directly: wind_farm_export.py --launch builds the MJCF and then starts this
    script, and it does not know about --ui. A child process inherits the
    environment, so exporting the variable switches the panel on either way.
    """
    if getattr(args, "ui", False):
        return True
    return os.environ.get("WIND_TURBINE_UI", "").strip().lower() in ("1", "true", "yes", "on")


def apply_environment(e, state, driver, announce=True):
    """Push one EnvironmentState snapshot into the driver and the semantic world.

    'state' is the local mirror of the last snapshot that was applied, so only
    the quantities that really changed are acted on: the wind is written to the
    wind-state file, the temperature is converted to an air density, and the
    grid limit is handed to the driver's curtailment logic. The snapshot may
    come from the viewer keys or from the browser panel -- this function does
    not care which, it only prints where it came from.
    """
    changed = []

    if e["wind_speed"] != state.get("speed") or e["wind_direction"] != state.get("direction"):
        state["speed"] = e["wind_speed"]
        state["direction"] = e["wind_direction"]
        # stashed on the driver so RosPublisher can read it, same as temp_c below
        driver.wind_speed = state["speed"]
        driver.wind_direction = state["direction"]
        write_wind_state(state["speed"], state["direction"])   # tell the semantic world
        changed.append(f"wind FROM {state['direction']:.1f} deg @ {state['speed']:.1f} m/s")

    if e["temperature"] != state.get("temp_c"):
        state["temp_c"] = e["temperature"]
        driver.rho = formulas.rho_for_temperature(state["temp_c"])
        driver.temp_c = state["temp_c"]
        changed.append(f"T = {state['temp_c']:.1f} degC -> rho = {driver.rho:.4f} kg/m^3")

    if e["grid_limit"] != state.get("gridlimit_mw"):
        state["gridlimit_mw"] = e["grid_limit"]
        if hasattr(driver, "grid_limit_w"):
            driver.grid_limit_w = state["gridlimit_mw"] * 1e6
        changed.append(f"grid limit = {state['gridlimit_mw']:.0f} MW")

    if announce and changed:
        print(f"  [{e['source']}] " + "   ".join(changed))
    return bool(changed)


def push_telemetry(env, driver, data):
    """Send the read-only numbers back to the browser panel (no-op without --ui)."""
    if env is None:
        return
    rpms = driver.rpm(data)
    spinning = sum(1 for v in rpms.values() if abs(v) > 1e-6)
    limit_w = getattr(driver, "grid_limit_w", None)
    available = getattr(driver, "last_available_power", 0.0)
    env.set_telemetry(
        power=getattr(driver, "last_total_power", 0.0) / 1e6,     # MW, same unit as the slider
        energy=getattr(driver, "total_energy_wh", 0.0) / 1000.0,  # kWh
        spinning=f"{spinning}/{len(rpms)}",
        curtailed=bool(limit_w is not None and available > limit_w + 1.0),
    )


# --------------------------------------------------------------------------- #
def run_headless(model, data, driver, seconds, wind_speed, publisher=None, direction=0.0,
                 history_path=None, env=None):
    steps = int(seconds / model.opt.timestep)
    pub_every = max(1, int(0.1 / model.opt.timestep))    # ~10 Hz
    hist_every = max(1, int(1.0 / model.opt.timestep))   # 1 Hz
    tele_every = max(1, int(0.2 / model.opt.timestep))   # 5 Hz, matches the panel's poll rate

    if env is None:
        env = EnvironmentState(wind_speed=wind_speed, wind_direction=direction % 360.0,
                               temperature=getattr(driver, "temp_c", DEFAULT_TEMP_C),
                               grid_limit=DEFAULT_GRIDLIMIT_MW)
    state = {}
    snap = env.snapshot()
    apply_environment(snap, state, driver, announce=False)
    last_version = snap["version"]
    wind = wind_from_bearing(state["speed"], state["direction"])

    print(f"Headless: {seconds}s @ {state['speed']} m/s (wind FROM {state['direction']:.1f} deg)\n")
    hist_n = 0
    for i in range(steps):
        snap = env.snapshot()
        if snap["version"] != last_version:          # keys or browser changed something
            last_version = snap["version"]
            apply_environment(snap, state, driver)
            wind = wind_from_bearing(state["speed"], state["direction"])
        driver.advance(data, wind, model.opt.timestep)
        if publisher and i % pub_every == 0:
            publisher.publish(driver)
        if i % tele_every == 0:
            push_telemetry(env, driver, data)
        if history_path and i % hist_every == 0:
            _write_history(driver, history_path, data.time, state["speed"], state["direction"])
            hist_n += 1
        if i % int(0.5 / model.opt.timestep) == 0:
            rpms = driver.rpm(data)
            txt = "  ".join(f"{k}={v:7.1f}rpm" for k, v in rpms.items())
            extra = f"   total={getattr(driver,'last_total_power',0)/1e6:6.2f}MW" \
                    f"  E={getattr(driver,'total_energy_wh',0)/1000:7.3f}kWh"
            print(f"t={data.time:5.2f}s   {txt}{extra}")
    if history_path:
        print(f"Recorded {hist_n} history samples to {history_path}")
    return driver.rpm(data)


def run_viewer(model, data, driver, wind_speed, publisher=None, direction=0.0, temp_c=DEFAULT_TEMP_C,
               run_seconds=None, history_path=None, env=None):
    import mujoco.viewer

    # The environment lives in one shared, thread-safe object. The arrow keys below
    # write into it, and so does the browser panel when --ui is on; the simulation
    # loop only ever reads it. Without --ui the object is still used, so there is a
    # single code path for both input methods.
    if env is None:
        limit = getattr(driver, "grid_limit_w", None)
        env = EnvironmentState(wind_speed=wind_speed, wind_direction=direction % 360.0,
                               temperature=temp_c,
                               grid_limit=limit / 1e6 if limit is not None else DEFAULT_GRIDLIMIT_MW)

    # mode: "speed"/"direction"/"temperature"/"gridlimit" -> which quantity the arrow keys control
    state = {"mode": "speed"}
    apply_environment(env.snapshot(), state, driver, announce=False)
    last_version = env.snapshot()["version"]

    def key_cb(keycode):
        if keycode in (83, 115):                     # 's' / 'S' -> speed mode
            state["mode"] = "speed"
            print("Mode: WIND SPEED (Up/Down arrows)")
            return
        if keycode in (68, 100):                      # 'd' / 'D' -> direction mode
            state["mode"] = "direction"
            print("Mode: WIND DIRECTION (Up/Down arrows, 0-360 deg)")
            return
        if keycode in (84, 116):                       # 't' / 'T' -> temperature mode
            state["mode"] = "temperature"
            print("Mode: TEMPERATURE (Up/Down arrows, deg C) -- changes air density (rho)")
            return
        if keycode in (71, 103):                       # 'g' / 'G' -> gridlimit mode
            state["mode"] = "gridlimit"
            print(f"Mode: GRID LIMIT (Up/Down arrows, +/- {GRIDLIMIT_STEP:.0f} MW) "
                  f"-- currently {env.grid_limit:.0f} MW")
            return
        if keycode in (82, 114):                       # 'r' / 'R' -> reset
            mujoco.mj_resetData(model, data)
            if hasattr(driver, "reset_energy"):
                driver.reset_energy()
            return
        if keycode == 265:                             # Up arrow
            sign = +1.0
        elif keycode == 264:                           # Down arrow
            sign = -1.0
        else:
            return

        # Only write the new value; clamping (>= 0 m/s) and the 0-360 wrap happen
        # inside EnvironmentState, and the effect is applied by the loop below.
        now = env.snapshot()
        if state["mode"] == "speed":
            env.update(source="keys", wind_speed=now["wind_speed"] + sign * SPEED_STEP)
        elif state["mode"] == "direction":
            env.update(source="keys", wind_direction=now["wind_direction"] + sign * DIR_STEP)
        elif state["mode"] == "temperature":
            env.update(source="keys", temperature=now["temperature"] + sign * TEMP_STEP)
        elif state["mode"] == "gridlimit":
            env.update(source="keys", grid_limit=now["grid_limit"] + sign * GRIDLIMIT_STEP)

    def current_wind():
        return wind_from_bearing(state["speed"], state["direction"])

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = 1.4 * model.stat.extent
        v.cam.elevation = -12
        v.cam.azimuth = 120
        start = time.time()
        print_t = pub_t = tele_t = start
        hist_t = start - 1.0        # so the first sample is written right away
        hist_n = 0
        while v.is_running():
            step_start = time.time()
            snap = env.snapshot()
            if snap["version"] != last_version:      # keys or browser changed something
                last_version = snap["version"]
                apply_environment(snap, state, driver)
            driver.advance(data, current_wind(), model.opt.timestep)
            v.sync()
            now = time.time()
            elapsed = now - start
            if publisher and now - pub_t > 0.1:             # ~10 Hz
                publisher.publish(driver)
                pub_t = now
            if now - tele_t > 0.2:                          # 5 Hz, feeds the browser readout
                push_telemetry(env, driver, data)
                tele_t = now
            if history_path and now - hist_t >= 1.0:        # 1 Hz history log
                _write_history(driver, history_path, elapsed, state["speed"], state["direction"])
                hist_n += 1
                hist_t = now
            if now - print_t > 1.0:
                rpms = driver.rpm(data)
                spinning = sum(1 for vv in rpms.values() if abs(vv) > 1e-6)
                extra = f"   total={getattr(driver,'last_total_power',0)/1e6:.2f}MW" \
                        f"  E={getattr(driver,'total_energy_wh',0)/1000:.3f}kWh"
                left = f"  ({run_seconds - elapsed:4.0f}s left)" if run_seconds else ""
                print(f"wind FROM {state['direction']:5.1f} deg @ {state['speed']:4.1f} m/s   "
                      f"T={state['temp_c']:4.1f}degC (rho={driver.rho:.3f})   "
                      f"spinning {spinning}/{len(rpms)}{extra}{left}")
                print_t = now
            if run_seconds is not None and elapsed >= run_seconds:
                print(f"\nReached --time {run_seconds:g}s, closing.")
                break
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    if history_path:
        print(f"Recorded {hist_n} history samples to {history_path}")


def main():
    ap = argparse.ArgumentParser(description="Wind turbines: query-driven (default) or aerodynamic (--aero)")
    ap.add_argument("--model", default="wind_turbine_generated.xml")
    ap.add_argument("--wind", type=float, default=8.0, help="wind speed m/s")
    ap.add_argument("--headless", type=float, default=None,
                    help="run N seconds headless instead of opening a viewer")
    ap.add_argument("--aero", action="store_true", help="use the aerodynamic force model")
    ap.add_argument("--axial", action="store_true",
                    help="query mode: use wind projected onto each rotor axis (orientation matters)")
    ap.add_argument("--direction", type=float, default=0.0,
                    help="wind comes FROM this compass bearing in degrees (0=N, 90=E, 180=S, "
                         "270=W). Only turbines facing it spin unless --all-spin is set. "
                         "Change live in the viewer: press 'd' then Up/Down arrows.")
    ap.add_argument("--all-spin", action="store_true",
                    help="disable directional gating: every turbine spins regardless of facing")
    ap.add_argument("--tsr", type=float, default=formulas.TSR, help="tip-speed ratio (default 6)")
    ap.add_argument("--gridLimit", type=float, default=None,
                    help="grid UPPER limit in MW; only run enough turbines so the total "
                         "generated power stays under it (curtailment). Default 500 MW in "
                         "the viewer; change it live with 'g' or with the browser panel.")
    ap.add_argument("--cp", type=float, default=None,
                    help="override cp with a fixed value; default is to use the "
                         "polynomial cp(v) fit (recommended, updates live with wind speed)")
    ap.add_argument("--yaw-rate", type=float, default=10.0,
                    help="nacelle yaw slew rate in deg/s while tracking the wind (default 10)")
    ap.add_argument("--rotor-accel", type=float, default=1.0,
                    help="max RPM change per second (rotor inertia): how fast the blades "
                         "speed up or slow down toward the wind-driven target (default 1.0)")
    ap.add_argument("--publish", action="store_true", help="publish rpm/power/energy to ROS 2 topics")
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP_C,
                    help="initial air temperature in deg C (default 15, -> rho=1.225 kg/m^3). "
                         "Sets rho via the ideal gas law. Change live in the viewer: "
                         "press 't' then Up/Down arrows.")
    ap.add_argument("--ui", action="store_true",
                    help="serve the browser control panel (sliders for wind speed, temperature "
                         "and grid limit, compass for wind direction) and open it. Can also be "
                         "switched on with WIND_TURBINE_UI=1, which is what wind_farm_export.py "
                         "--launch passes through.")
    ap.add_argument("--ui-port", type=int, default=int(os.environ.get("WIND_TURBINE_UI_PORT", 8080)),
                    help="port for the control panel (default 8080, or WIND_TURBINE_UI_PORT)")
    ap.add_argument("--grid-limit-max", type=float, default=1000.0,
                    help="upper end of the grid-limit range in MW (default 1000). Raised "
                         "automatically if --gridLimit is larger, so the value passed is "
                         "never clamped away.")
    ap.add_argument("--time", type=float, default=None,
                    help="run for N seconds then close automatically, logging history at 1 Hz")
    ap.add_argument("--history-file", default="history.jsonl",
                    help="path for the 1 Hz history log (used with --time)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    initial_rho = formulas.rho_for_temperature(args.temp)

    if args.aero:
        driver = Aerodynamics(model)
        driver.rho = initial_rho
        driver.temp_c = args.temp
        print(f"AERO model: {len(driver.blades)} blades on {len(driver.rotors)} rotors.")
    else:
        facing = not (args.axial or args.all_spin)
        driver = QueryDriver(model, axial=args.axial, tsr=args.tsr,
                             grid_limit_mw=args.gridLimit, rho=initial_rho, c_p=args.cp, facing=facing,
                             yaw_rate_deg=args.yaw_rate, rotor_accel_rpm_s=args.rotor_accel)
        driver.temp_c = args.temp
        mode = (f"wind FROM {args.direction:.1f} deg, facing-gated" if facing
                else "axial wind" if args.axial else "scalar wind (all spin)")
        print(f"QUERY model ({mode}, TSR={args.tsr:g}):")
        for name, D, vmin in driver.info():
            print(f"  {name}: blade D={D:.3f} m -> cut-in {vmin:.4f} m/s (1 RPM)")
        if args.gridLimit is not None:
            preview_wind = wind_from_bearing(args.wind, args.direction)
            _, used, total, on = driver.preview(data, preview_wind)
            print(f"\nGrid limit {args.gridLimit:g} MW @ {args.wind:g} m/s: "
                  f"running {len(on)}/{len(driver.rotors)} turbines "
                  f"({used/1e6:.3f} MW used of {total/1e6:.3f} MW available)")
            running = set(on)
            for name, *_ in driver.rotors:
                print(f"    {name:28s} {'RUN ' if name in running else 'idle'}")

    # Widen the grid-limit range BEFORE building the state, otherwise a --gridLimit
    # above the range maximum would be clamped away without a word.
    limit_max = max(args.grid_limit_max, args.gridLimit or 0.0, DEFAULT_GRIDLIMIT_MW)
    set_field_range("grid_limit", maximum=limit_max, step=max(1.0, round(limit_max / 100.0)))

    # One shared environment for the viewer keys and the browser panel.
    env = EnvironmentState(
        wind_speed=args.wind,
        wind_direction=args.direction % 360.0,
        temperature=args.temp,
        grid_limit=args.gridLimit if args.gridLimit is not None else DEFAULT_GRIDLIMIT_MW,
    )
    if ui_requested(args):
        start_control_server(env, port=args.ui_port, open_browser=True)
    clear_peak_state()

    publisher = None
    if args.publish:
        rotor_names = [r[0] for r in driver.rotors]
        try:
            publisher = RosPublisher(rotor_names)
            print(f"\nPublishing {len(rotor_names)} turbines on /wind_farm/* (std_msgs/Float64)")
        except Exception as e:  # noqa: BLE001
            print(f"\n[publish disabled] could not start ROS 2 publisher: {e}")

    history_path = None
    if args.time is not None:
        if _HISTORY_OK:
            history_path = args.history_file
            clear_history(history_path)       # fresh file for this run
            print(f"Recording history at 1 Hz to {history_path} for {args.time:g}s")
        else:
            print("[history disabled] history_file.py not importable; running without a log")

    try:
        if args.headless is not None:
            run_headless(model, data, driver, args.headless, args.wind, publisher,
                         args.direction, history_path, env=env)
        else:
            run_viewer(model, data, driver, args.wind, publisher, args.direction, args.temp,
                       run_seconds=args.time, history_path=history_path, env=env)
    finally:
        if publisher:
            publisher.shutdown()


if __name__ == "__main__":
    main()