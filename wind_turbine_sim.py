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
    R = reset (also zeros energy)
Each turbine's nacelle automatically yaws to face wherever the wind is currently
coming from, slewing smoothly rather than snapping to the new heading. Rotor RPM
and generated power likewise ramp up/down toward their wind-driven target instead
of snapping there instantly.
===================================================================
"""

import argparse
import time
from datetime import datetime
import numpy as np
import mujoco

import turbine_formulas as formulas  # pure cut-in + RPM + power formulas (no framework)
from wind_state_file import write_wind_state  # shares live wind with main1.py's semantic world

try:
    from history_file import append_sample, clear_history   # optional 1 Hz history logging
    _HISTORY_OK = True
except Exception:  # noqa: BLE001
    _HISTORY_OK = False

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
                 needed_mw: float = None, rho: float = formulas.RHO, c_p: float = None,
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
        self.needed_w = None if needed_mw is None else needed_mw * 1e6
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

        self.names = [r[0] for r in self.rotors]
        self.energy_wh = {n: 0.0 for n in self.names}    # cumulative per turbine
        self.total_energy_wh = 0.0
        self.rotor_rpm = {k: 0.0 for k in range(len(self.rotors))}   # actual (ramped) RPM per rotor
        self.last_readings = {}                          # name -> (rpm, power_w, energy_wh, active, cp)
        self.last_total_power = 0.0

    def _blade_length(self, hub_body_id):
        for g in range(self.model.ngeom):
            gb = self.model.geom_bodyid[g]
            if self.model.body_parentid[gb] != hub_body_id:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.endswith("_geom"):
                return 2.0 * float(self.model.geom_size[g][2])
        return 1.0

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
            power = formulas.generated_power_for_length(self.rho, v, D, self.c_p) if rpm != 0 else 0.0
            rows.append((k, rpm, power, cp))

        if self.needed_w is None:
            return {k for k, rpm, _, _ in rows if rpm != 0}, rows

        active, cum = set(), 0.0
        for k, rpm, power, cp in sorted(rows, key=lambda r: (-r[2], self.rotors[r[0]][0])):
            if cum >= self.needed_w:
                break
            if power <= 0:
                continue
            active.add(k)
            cum += power
        return active, rows

    def advance(self, data: mujoco.MjData, wind_vec: np.ndarray, dt: float):
        self._update_yaw(data, wind_vec, dt)
        active, rows = self._active_set(data, wind_vec)
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
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

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
    sample = {
        "time_s": float(elapsed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wind_speed": float(wind_speed),
        "wind_direction_deg": float(direction_deg),
        "total_power_w": float(getattr(driver, "last_total_power", 0.0)),
        "rpm": {base(n): r[0] for n, r in driver.last_readings.items()},
        "power_w": {base(n): r[1] for n, r in driver.last_readings.items()},
    }
    try:
        append_sample(sample, path)
    except Exception as exc:   # noqa: BLE001
        print(f"[history] write failed: {exc!r}")


# --------------------------------------------------------------------------- #
def run_headless(model, data, driver, seconds, wind_speed, publisher=None, direction=0.0,
                 history_path=None):
    steps = int(seconds / model.opt.timestep)
    pub_every = max(1, int(0.1 / model.opt.timestep))    # ~10 Hz
    hist_every = max(1, int(1.0 / model.opt.timestep))   # 1 Hz
    wind = wind_from_bearing(wind_speed, direction)
    write_wind_state(wind_speed, direction)   # let the semantic world see this run's wind too
    print(f"Headless: {seconds}s @ {wind_speed} m/s (wind FROM {direction:.1f} deg)\n")
    hist_n = 0
    for i in range(steps):
        driver.advance(data, wind, model.opt.timestep)
        if publisher and i % pub_every == 0:
            publisher.publish(driver)
        if history_path and i % hist_every == 0:
            _write_history(driver, history_path, data.time, wind_speed, direction)
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


SPEED_STEP = 1.0    # m/s per arrow press
DIR_STEP = 5.0       # degrees per arrow press
TEMP_STEP = 1.0      # deg C per arrow press
DEFAULT_TEMP_C = 15.0   # matches RHO = 1.225 kg/m^3 (ENERCON's "Standardluftdichte")


def run_viewer(model, data, driver, wind_speed, publisher=None, direction=0.0, temp_c=DEFAULT_TEMP_C,
               run_seconds=None, history_path=None):
    import mujoco.viewer
    # mode: "speed"/"direction"/"temperature" -> which quantity the arrow keys control
    state = {"speed": wind_speed, "direction": direction % 360.0, "mode": "speed",
             "temp_c": temp_c}
    write_wind_state(state["speed"], state["direction"])  # publish initial wind immediately
    driver.rho = formulas.rho_for_temperature(state["temp_c"])
    driver.temp_c = state["temp_c"]

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
        if keycode in (82, 114):                       # 'r' / 'R' -> reset
            mujoco.mj_resetData(model, data)
            if hasattr(driver, "reset_energy"):
                driver.reset_energy()
            return
        if keycode == 265:                             # Up arrow
            if state["mode"] == "speed":
                state["speed"] += SPEED_STEP
            elif state["mode"] == "direction":
                state["direction"] = (state["direction"] + DIR_STEP) % 360.0
            else:
                state["temp_c"] += TEMP_STEP
        elif keycode == 264:                           # Down arrow
            if state["mode"] == "speed":
                state["speed"] = max(0.0, state["speed"] - SPEED_STEP)
            elif state["mode"] == "direction":
                state["direction"] = (state["direction"] - DIR_STEP) % 360.0
            else:
                state["temp_c"] -= TEMP_STEP
        else:
            return
        if state["mode"] == "temperature":
            driver.rho = formulas.rho_for_temperature(state["temp_c"])
            driver.temp_c = state["temp_c"]
            print(f"  temperature = {state['temp_c']:.1f} degC   "
                  f"-> rho = {driver.rho:.4f} kg/m^3")
            return
        write_wind_state(state["speed"], state["direction"])  # tell the semantic world
        if state["mode"] == "speed":
            print(f"  wind = {state['speed']:.1f} m/s")
        else:
            print(f"  wind FROM {state['direction']:.1f} deg")

    def current_wind():
        return wind_from_bearing(state["speed"], state["direction"])

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = 1.4 * model.stat.extent
        v.cam.elevation = -12
        v.cam.azimuth = 120
        start = time.time()
        print_t = pub_t = start
        hist_t = start - 1.0        # so the first sample is written right away
        hist_n = 0
        while v.is_running():
            step_start = time.time()
            driver.advance(data, current_wind(), model.opt.timestep)
            v.sync()
            now = time.time()
            elapsed = now - start
            if publisher and now - pub_t > 0.1:             # ~10 Hz
                publisher.publish(driver)
                pub_t = now
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
    ap.add_argument("--needed", type=float, default=None,
                    help="required power output in MW; only enough turbines spin to meet it")
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
                             needed_mw=args.needed, rho=initial_rho, c_p=args.cp, facing=facing,
                             yaw_rate_deg=args.yaw_rate, rotor_accel_rpm_s=args.rotor_accel)
        driver.temp_c = args.temp
        mode = (f"wind FROM {args.direction:.1f} deg, facing-gated" if facing
                else "axial wind" if args.axial else "scalar wind (all spin)")
        print(f"QUERY model ({mode}, TSR={args.tsr:g}):")
        for name, D, vmin in driver.info():
            print(f"  {name}: blade D={D:.3f} m -> cut-in {vmin:.4f} m/s (1 RPM)")
        if args.needed is not None:
            preview_wind = wind_from_bearing(args.wind, args.direction)
            _, used, total, on = driver.preview(data, preview_wind)
            print(f"\nDemand {args.needed:g} MW @ {args.wind:g} m/s: "
                  f"starting {len(on)}/{len(driver.rotors)} turbines "
                  f"({used/1e6:.3f} MW of {total/1e6:.3f} MW available)")
            running = set(on)
            for name, *_ in driver.rotors:
                print(f"    {name:28s} {'RUN ' if name in running else 'idle'}")

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
                         args.direction, history_path)
        else:
            run_viewer(model, data, driver, args.wind, publisher, args.direction, args.temp,
                       run_seconds=args.time, history_path=history_path)
    finally:
        if publisher:
            publisher.shutdown()


if __name__ == "__main__":
    main()