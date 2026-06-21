"""
wind_turbine_sim.py
===================================================================
Drive the MuJoCo wind turbines.

Two drivers are available:

  QUERY MODE (default) -- queries.py
      Each rotor's RPM is set directly from the semantic query:
          RPM = (60 * v * 6) / (pi * D)        v = wind speed, D = blade length
      Below the cut-in speed  v_min = (pi * D) / 360  (the 1-RPM speed,
      i.e. queries.minimum_wind_speed) the rotor is HELD STILL.
      The motion is prescribed kinematically (no aero forces).

  AERO MODE (--aero) -- the original force model
      Wind force is applied to the pitched blades and the rotor spins
      up freely to a steady tip speed.

Run:
    python wind_turbine_sim.py                       # query mode, 8 m/s, viewer
    python wind_turbine_sim.py --wind 0.01           # below cut-in -> no spin
    python wind_turbine_sim.py --axial               # wind projected onto each rotor axis
    python wind_turbine_sim.py --headless 5          # 5 s headless, prints RPM
    python wind_turbine_sim.py --aero                # original aerodynamic model

Viewer keys:
    Up / Down arrow : wind speed +/- 1 m/s
    R               : reset rotors to rest
===================================================================
"""

import argparse
import time
import numpy as np
import mujoco

import turbine_formulas as formulas  # pure cut-in + RPM formulas (no framework)

RHO = 1.225   # air density [kg/m^3]   (aero mode only)
CD  = 1.28    # flat-plate drag coeff  (aero mode only)


# =================================================================== #
# QUERY-DRIVEN KINEMATIC MODEL (default)
# =================================================================== #
class QueryDriver:
    """Set each rotor's RPM from formulas.rpm_for_wind, gated by the cut-in speed."""

    def __init__(self, model: mujoco.MjModel, axial: bool = False, tsr: float = formulas.TSR):
        self.model = model
        self.axial = axial          # project wind onto each rotor axis if True
        self.tsr = tsr              # tip-speed ratio
        self.rotors = []            # (name, body_id, qpos_adr, dof_adr, axis_local, D)

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name or not name.endswith("_rotor"):
                continue
            body_id = model.jnt_bodyid[j]
            qpadr   = model.jnt_qposadr[j]
            dadr    = model.jnt_dofadr[j]
            axis    = model.jnt_axis[j].copy()
            D       = self._blade_length(body_id)        # full blade length
            self.rotors.append((name, body_id, qpadr, dadr, axis, D))

        if not self.rotors:
            raise RuntimeError("No '*_rotor' joints found in the model.")

    def _blade_length(self, hub_body_id):
        """Blade length D = visual.scale.z = 2 * (blade '_geom' half-size z)."""
        for g in range(self.model.ngeom):
            gb = self.model.geom_bodyid[g]
            # the main blade geom sits on a body whose parent is the hub
            if self.model.body_parentid[gb] != hub_body_id:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.endswith("_geom"):
                return 2.0 * float(self.model.geom_size[g][2])
        return 1.0

    def advance(self, data: mujoco.MjData, wind_vec: np.ndarray, dt: float):
        for name, body_id, qpadr, dadr, axis_local, D in self.rotors:
            if self.axial:
                R = data.xmat[body_id].reshape(3, 3)
                axis_world = R @ axis_local
                v = float(np.dot(wind_vec, axis_world))   # signed axial wind
            else:
                v = float(np.linalg.norm(wind_vec))       # plain wind speed

            rpm = formulas.rpm_for_wind(v, D, self.tsr)    # 0 below cut-in
            omega = rpm * 2.0 * np.pi / 60.0              # rad/s
            data.qvel[dadr] = omega
            data.qpos[qpadr] += omega * dt                # prescribe rotation
        mujoco.mj_forward(self.model, data)               # kinematics only
        data.time += dt

    def rpm(self, data):
        return {name: float(data.qvel[dadr]) * 60.0 / (2 * np.pi)
                for name, _, _, dadr, _, _ in self.rotors}

    def info(self):
        return [(name, D, formulas.min_wind_speed_for_length(D, self.tsr))
                for name, _, _, _, _, D in self.rotors]


# =================================================================== #
# ORIGINAL AERODYNAMIC FORCE MODEL  (--aero)
# =================================================================== #
class Aerodynamics:
    """Applies wind force to every blade aero site each step."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.blades = []

        for s in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)
            if name is None or not name.endswith("_aero"):
                continue
            body_id = model.site_bodyid[s]
            area = self._plate_area(body_id)
            self.blades.append((s, body_id, area))

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

    def apply(self, data: mujoco.MjData, wind_vec: np.ndarray):
        data.xfrc_applied[:] = 0.0
        vel6 = np.zeros(6)
        for site_id, body_id, area in self.blades:
            p = data.site_xpos[site_id]
            n = data.site_xmat[site_id].reshape(3, 3)[:, 0]
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel6, 0)
            v_point = vel6[3:6]
            v_rel = wind_vec - v_point
            v_n = float(np.dot(v_rel, n))
            F = 0.5 * RHO * CD * area * v_n * abs(v_n) * n
            r = p - data.xipos[body_id]
            data.xfrc_applied[body_id, 0:3] += F
            data.xfrc_applied[body_id, 3:6] += np.cross(r, F)

    def advance(self, data: mujoco.MjData, wind_vec: np.ndarray, dt: float):
        self.apply(data, wind_vec)
        mujoco.mj_step(self.model, data)

    def rpm(self, data):
        return {name: float(data.qvel[adr]) * 60.0 / (2 * np.pi)
                for name, adr in self.rotors}


def make_wind(speed, heading_deg=0.0):
    """Horizontal wind vector. heading 0 deg -> blowing along +X."""
    a = np.deg2rad(heading_deg)
    return np.array([np.cos(a), np.sin(a), 0.0]) * speed


# --------------------------------------------------------------------------- #
def run_headless(model, data, driver, seconds, wind_speed):
    steps = int(seconds / model.opt.timestep)
    wind = make_wind(wind_speed)
    print(f"Headless: {seconds}s @ {wind_speed} m/s wind\n")
    for i in range(steps):
        driver.advance(data, wind, model.opt.timestep)
        if i % int(0.5 / model.opt.timestep) == 0:
            rpms = driver.rpm(data)
            txt = "  ".join(f"{k}={v:7.1f}rpm" for k, v in rpms.items())
            print(f"t={data.time:5.2f}s   {txt}")
    return driver.rpm(data)


def run_viewer(model, data, driver, wind_speed):
    import mujoco.viewer

    state = {"speed": wind_speed, "heading": 0.0}

    def key_cb(keycode):
        if keycode == 265:
            state["speed"] += 1.0
        elif keycode == 264:
            state["speed"] = max(0.0, state["speed"] - 1.0)
        elif keycode in (82, 114):
            mujoco.mj_resetData(model, data)
        print(f"  wind = {state['speed']:.1f} m/s")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        # frame the whole scene (works for one turbine or many farms)
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = 1.4 * model.stat.extent
        v.cam.elevation = -12
        v.cam.azimuth = 120
        print_t = time.time()
        while v.is_running():
            step_start = time.time()
            wind = make_wind(state["speed"], state["heading"])
            driver.advance(data, wind, model.opt.timestep)
            v.sync()
            if time.time() - print_t > 1.0:
                rpms = driver.rpm(data)
                txt = "  ".join(f"{k}={v_:7.1f}rpm" for k, v_ in rpms.items())
                print(f"wind={state['speed']:5.1f} m/s   {txt}")
                print_t = time.time()
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
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    if args.aero:
        driver = Aerodynamics(model)
        print(f"AERO model: {len(driver.blades)} blades on {len(driver.rotors)} rotors.")
    else:
        driver = QueryDriver(model, axial=args.axial, tsr=args.tsr)
        print(f"QUERY model ({'axial wind' if args.axial else 'scalar wind'}, TSR={args.tsr:g}):")
        for name, D, vmin in driver.info():
            print(f"  {name}: blade D={D:.3f} m -> cut-in {vmin:.4f} m/s (1 RPM)")

    if args.headless is not None:
        run_headless(model, data, driver, args.headless, args.wind)
    else:
        run_viewer(model, data, driver, args.wind)


if __name__ == "__main__":
    main()