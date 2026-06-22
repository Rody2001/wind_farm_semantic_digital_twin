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
    /wind_farm/total_power_w
    /wind_farm/total_energy_kwh

Other options:
    --wind 8 --tsr 6 --cp 0.45 --axial --aero --headless N
Viewer keys: Up/Down = wind +/- 1 m/s,  R = reset (also zeros energy)
===================================================================
"""

import argparse
import time
import numpy as np
import mujoco

import turbine_formulas as formulas  # pure cut-in + RPM + power formulas (no framework)

RHO = formulas.RHO   # air density [kg/m^3]
CD  = 1.28           # flat-plate drag coeff  (aero mode only)


# =================================================================== #
# QUERY-DRIVEN KINEMATIC MODEL (default)
# =================================================================== #
class QueryDriver:
    """RPM from formulas.rpm_for_wind, gated by cut-in and an optional demand cap.
    Tracks generated power and accumulated energy for publishing."""

    def __init__(self, model: mujoco.MjModel, axial: bool = False, tsr: float = formulas.TSR,
                 needed_mw: float = None, rho: float = formulas.RHO, c_p: float = formulas.C_P):
        self.model = model
        self.axial = axial
        self.tsr = tsr
        self.rho = rho
        self.c_p = c_p
        self.needed_w = None if needed_mw is None else needed_mw * 1e6
        self.rotors = []            # (name, body_id, qpos_adr, dof_adr, axis_local, D)

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name or not name.endswith("_rotor"):
                continue
            body_id = model.jnt_bodyid[j]
            qpadr   = model.jnt_qposadr[j]
            dadr    = model.jnt_dofadr[j]
            axis    = model.jnt_axis[j].copy()
            D       = self._blade_length(body_id)
            self.rotors.append((name, body_id, qpadr, dadr, axis, D))

        if not self.rotors:
            raise RuntimeError("No '*_rotor' joints found in the model.")

        self.names = [r[0] for r in self.rotors]
        self.energy_wh = {n: 0.0 for n in self.names}    # cumulative per turbine
        self.total_energy_wh = 0.0
        self.last_readings = {}                          # name -> (rpm, power_w, energy_wh, active)
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

    def _rotor_wind(self, data, body_id, axis_local, wind_vec):
        if self.axial:
            R = data.xmat[body_id].reshape(3, 3)
            return float(np.dot(wind_vec, R @ axis_local))
        return float(np.linalg.norm(wind_vec))

    def _active_set(self, data, wind_vec):
        rows = []
        for k, (name, body_id, _, _, axis_local, D) in enumerate(self.rotors):
            v = self._rotor_wind(data, body_id, axis_local, wind_vec)
            rpm = formulas.rpm_for_wind(v, D, self.tsr)
            power = formulas.generated_power_for_length(self.rho, v, D, self.c_p) if rpm != 0 else 0.0
            rows.append((k, rpm, power))

        if self.needed_w is None:
            return {k for k, rpm, _ in rows if rpm != 0}, rows

        active, cum = set(), 0.0
        for k, rpm, power in sorted(rows, key=lambda r: (-r[2], self.rotors[r[0]][0])):
            if cum >= self.needed_w:
                break
            if power <= 0:
                continue
            active.add(k)
            cum += power
        return active, rows

    def advance(self, data: mujoco.MjData, wind_vec: np.ndarray, dt: float):
        active, rows = self._active_set(data, wind_vec)
        total_power = 0.0
        for k, rpm, power in rows:
            name, body_id, qpadr, dadr, axis_local, D = self.rotors[k]
            is_on = k in active
            omega = (rpm * 2.0 * np.pi / 60.0) if is_on else 0.0
            data.qvel[dadr] = omega
            data.qpos[qpadr] += omega * dt
            gen = power if is_on else 0.0                 # idle turbines make nothing
            self.energy_wh[name] += gen * dt / 3600.0
            self.last_readings[name] = (omega * 60.0 / (2 * np.pi), gen,
                                        self.energy_wh[name], is_on)
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
        used = sum(p for k, _, p in rows if k in active)
        total = sum(p for _, _, p in rows)
        names_on = [self.rotors[k][0] for k, _, _ in rows if k in active]
        return active, used, total, names_on


# =================================================================== #
# ORIGINAL AERODYNAMIC FORCE MODEL  (--aero)
# =================================================================== #
class Aerodynamics:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
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
            F = 0.5 * RHO * CD * area * v_n * abs(v_n) * n
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
        self.rpm_pubs, self.power_pubs, self.energy_pubs = {}, {}, {}
        for n in rotor_names:
            base = n[:-6] if n.endswith("_rotor") else n     # strip "_rotor"
            self.rpm_pubs[n]    = self.node.create_publisher(Float64, f"/wind_farm/{base}/rpm", 10)
            self.power_pubs[n]  = self.node.create_publisher(Float64, f"/wind_farm/{base}/power_w", 10)
            self.energy_pubs[n] = self.node.create_publisher(Float64, f"/wind_farm/{base}/energy_kwh", 10)
        self.total_power_pub  = self.node.create_publisher(Float64, "/wind_farm/total_power_w", 10)
        self.total_energy_pub = self.node.create_publisher(Float64, "/wind_farm/total_energy_kwh", 10)

    def publish(self, driver):
        F = self._Float64
        for name, (rpm, power, energy_wh, _active) in driver.last_readings.items():
            self.rpm_pubs[name].publish(F(data=float(rpm)))
            self.power_pubs[name].publish(F(data=float(power)))
            self.energy_pubs[name].publish(F(data=float(energy_wh / 1000.0)))
        self.total_power_pub.publish(F(data=float(driver.last_total_power)))
        self.total_energy_pub.publish(F(data=float(driver.total_energy_wh / 1000.0)))
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def make_wind(speed, heading_deg=0.0):
    a = np.deg2rad(heading_deg)
    return np.array([np.cos(a), np.sin(a), 0.0]) * speed


# --------------------------------------------------------------------------- #
def run_headless(model, data, driver, seconds, wind_speed, publisher=None):
    steps = int(seconds / model.opt.timestep)
    pub_every = max(1, int(0.1 / model.opt.timestep))    # ~10 Hz
    wind = make_wind(wind_speed)
    print(f"Headless: {seconds}s @ {wind_speed} m/s wind\n")
    for i in range(steps):
        driver.advance(data, wind, model.opt.timestep)
        if publisher and i % pub_every == 0:
            publisher.publish(driver)
        if i % int(0.5 / model.opt.timestep) == 0:
            rpms = driver.rpm(data)
            txt = "  ".join(f"{k}={v:7.1f}rpm" for k, v in rpms.items())
            extra = f"   total={getattr(driver,'last_total_power',0)/1e6:6.2f}MW" \
                    f"  E={getattr(driver,'total_energy_wh',0)/1000:7.3f}kWh"
            print(f"t={data.time:5.2f}s   {txt}{extra}")
    return driver.rpm(data)


def run_viewer(model, data, driver, wind_speed, publisher=None):
    import mujoco.viewer
    state = {"speed": wind_speed, "heading": 0.0}

    def key_cb(keycode):
        if keycode == 265:
            state["speed"] += 1.0
        elif keycode == 264:
            state["speed"] = max(0.0, state["speed"] - 1.0)
        elif keycode in (82, 114):
            mujoco.mj_resetData(model, data)
            if hasattr(driver, "reset_energy"):
                driver.reset_energy()
        print(f"  wind = {state['speed']:.1f} m/s")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = 1.4 * model.stat.extent
        v.cam.elevation = -12
        v.cam.azimuth = 120
        print_t = pub_t = time.time()
        while v.is_running():
            step_start = time.time()
            wind = make_wind(state["speed"], state["heading"])
            driver.advance(data, wind, model.opt.timestep)
            v.sync()
            now = time.time()
            if publisher and now - pub_t > 0.1:             # ~10 Hz
                publisher.publish(driver)
                pub_t = now
            if now - print_t > 1.0:
                rpms = driver.rpm(data)
                txt = "  ".join(f"{k}={v_:7.1f}rpm" for k, v_ in rpms.items())
                extra = f"   total={getattr(driver,'last_total_power',0)/1e6:.2f}MW" \
                        f"  E={getattr(driver,'total_energy_wh',0)/1000:.3f}kWh"
                print(f"wind={state['speed']:5.1f} m/s   {txt}{extra}")
                print_t = now
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main():
    ap = argparse.ArgumentParser(description="Wind turbines: query-driven (default) or aerodynamic (--aero)")
    ap.add_argument("--model", default="wind_turbine_generated.xml")
    ap.add_argument("--wind", type=float, default=8.0, help="wind speed m/s")
    ap.add_argument("--headless", type=float, default=None,
                    help="run N seconds headless instead of opening a viewer")
    ap.add_argument("--aero", action="store_true", help="use the aerodynamic force model")
    ap.add_argument("--axial", action="store_true",
                    help="query mode: use wind projected onto each rotor axis (orientation matters)")
    ap.add_argument("--tsr", type=float, default=formulas.TSR, help="tip-speed ratio (default 6)")
    ap.add_argument("--needed", type=float, default=None,
                    help="required power output in MW; only enough turbines spin to meet it")
    ap.add_argument("--cp", type=float, default=formulas.C_P, help="power coefficient c_p (default 0.45)")
    ap.add_argument("--publish", action="store_true", help="publish rpm/power/energy to ROS 2 topics")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.aero:
        driver = Aerodynamics(model)
        print(f"AERO model: {len(driver.blades)} blades on {len(driver.rotors)} rotors.")
    else:
        driver = QueryDriver(model, axial=args.axial, tsr=args.tsr,
                             needed_mw=args.needed, rho=RHO, c_p=args.cp)
        print(f"QUERY model ({'axial wind' if args.axial else 'scalar wind'}, TSR={args.tsr:g}):")
        for name, D, vmin in driver.info():
            print(f"  {name}: blade D={D:.3f} m -> cut-in {vmin:.4f} m/s (1 RPM)")
        if args.needed is not None:
            _, used, total, on = driver.preview(data, make_wind(args.wind))
            print(f"\nDemand {args.needed:g} MW @ {args.wind:g} m/s: "
                  f"starting {len(on)}/{len(driver.rotors)} turbines "
                  f"({used/1e6:.3f} MW of {total/1e6:.3f} MW available)")
            running = set(on)
            for name, *_ in driver.rotors:
                print(f"    {name:24s} {'RUN ' if name in running else 'idle'}")

    publisher = None
    if args.publish:
        rotor_names = [r[0] for r in driver.rotors]
        try:
            publisher = RosPublisher(rotor_names)
            print(f"\nPublishing {len(rotor_names)} turbines on /wind_farm/* (std_msgs/Float64)")
        except Exception as e:  # noqa: BLE001
            print(f"\n[publish disabled] could not start ROS 2 publisher: {e}")

    try:
        if args.headless is not None:
            run_headless(model, data, driver, args.headless, args.wind, publisher)
        else:
            run_viewer(model, data, driver, args.wind, publisher)
    finally:
        if publisher:
            publisher.shutdown()


if __name__ == "__main__":
    main()