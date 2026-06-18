"""
wind_farm_export.py
===================================================================
ONE-BUTTON EXPORT:  turn the wind farm defined in `main.py` into a
MuJoCo XML scene that `wind_turbine_sim.py` can drive directly.

The farm is described once, here, as a list of `TurbineSpec`s (the
single source of truth). `main.py` imports this list to build its
Semantic-Digital-Twin world, and this module turns the *same* list
into the `.xml` file the MuJoCo aerodynamic sim expects.

Geometry note
-------------
Every dimension below is derived from the EXACT same formulas
`WindTurbine.create_with_new_body_in_world` uses in main.py
(tower / nacelle / blade sizes are all functions of `tower_height`).
The rotor + blade *layout* follows wind_turbine(1).xml: three blades
at 120 deg spacing, each pitched by PITCH degrees about its span so
axial wind produces torque, with a `*_aero` site and `*_rotor` hinge
the sim looks for. That makes the output both faithful to main.py and
actually spin-able by the sim.

Run (the button):
    python wind_farm_export.py                 # writes wind_turbine_generated.xml
    python wind_farm_export.py --launch        # write, then open the sim viewer
    python wind_farm_export.py --launch --wind 12
    python wind_farm_export.py --out farm.xml
===================================================================
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Blade pitch (cant) about the span, in degrees. This is what turns
# axial wind into rotor torque. Matches wind_turbine(1).xml.
PITCH_DEG = 18.0

# main.py scale ratios (all relative to tower_height). Kept identical so
# the exported geometry matches the digital-twin model exactly.
R_TOWER_WIDTH   = 901  / 15797   # cylinder *diameter* in main.py
R_NACELLE_LEN   = 1500 / 15797
R_NACELLE_HEIGHT= 678  / 15797
R_BLADE_LENGTH  = 8475 / 15797
R_BLADE_X       = 0.04 / 5
R_BLADE_Y       = 0.18 / 5


@dataclass
class TurbineSpec:
    """One turbine in the farm. Mirrors a WindTurbine.create_* call."""
    name: str
    tower_height: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.1
    yaw: float = 0.0                       # radians, about world Z
    rotor_blade_length: float = 0.0        # 0 -> derived from tower_height


# ===================================================================
# SINGLE SOURCE OF TRUTH
# Edit this list to change the farm. main.py builds from it too.
# (These three entries reproduce the current main.py exactly.)
# ===================================================================
WIND_FARM: list[TurbineSpec] = [
    TurbineSpec("turbine1", tower_height=1.0,  x=0,  y=0, z=0.1, yaw=math.pi),
    TurbineSpec("turbine2", tower_height=3.0,  x=5,  y=5, z=0.1),
    TurbineSpec("turbine3", tower_height=10.0, x=-5, y=5, z=0.1, yaw=math.pi / 2),
    TurbineSpec("turbine4", tower_height=100.0, x=5,  z=0.1, yaw=math.pi / 2),
    TurbineSpec("turbine5", tower_height=200.0, x=50, z=0.1, yaw=math.pi / 2),
]


# ------------------------------------------------------------------- #
# XML helpers
# ------------------------------------------------------------------- #
def _f(v: float) -> str:
    """Compact float formatting."""
    return f"{v:.6g}"


def _turbine_body(spec: TurbineSpec) -> str:
    H = spec.tower_height
    n = spec.name

    # --- dimensions (identical to main.py formulas) ---
    tower_width   = H * R_TOWER_WIDTH          # diameter
    tower_radius  = tower_width / 2.0
    nacelle_len   = H * R_NACELLE_LEN
    nacelle_h     = H * R_NACELLE_HEIGHT
    blade_x_half  = (H * R_BLADE_X) / 2.0
    blade_y_half  = (H * R_BLADE_Y) / 2.0
    L = spec.rotor_blade_length or (H * R_BLADE_LENGTH)   # full blade length

    base_half     = H / 6.0                    # tower-base box half size
    yaw_deg       = math.degrees(spec.yaw)

    # --- vertical layout (base frame) ---
    tower_z   = H / 2.0 + 0.1
    nacelle_z = H + 0.1 + nacelle_h / 2.0
    hub_z     = nacelle_z
    hub_x     = nacelle_len / 4.0 + tower_radius

    # hub cosmetics (no effect on physics) ~ example ratios
    hub_r     = 0.03 * H
    hub_hl    = 0.02 * H
    spin_a, spin_b = 0.05 * H, 0.028 * H

    # --- blade layout ---
    blade_z   = L / 2.0 + nacelle_h / 2.0      # main blade-box centre
    tip_half  = 0.03 * L
    tip_z     = blade_z + L / 2.0 - tip_half
    aero_z    = blade_z + 0.055 * L
    aero_r    = 0.006 * H
    radials   = (-120.0, 120.0, 0.0)           # 3-blade fan

    def blade(i: int, rx: float) -> str:
        b = f"{n}_blade{i}"
        return f"""\
        <body name="{b}" euler="{_f(rx)} 0 {_f(-PITCH_DEG)}">
          <geom name="{b}_geom" type="box" size="{_f(blade_x_half)} {_f(blade_y_half)} {_f(L/2)}" pos="0 0 {_f(blade_z)}" material="blade_mat"/>
          <geom name="{b}_tip" type="box" size="{_f(blade_x_half)} {_f(blade_y_half)} {_f(tip_half)}" pos="0 0 {_f(tip_z)}" material="tip_mat"/>
          <site name="{b}_aero" pos="0 0 {_f(aero_z)}" size="{_f(aero_r)}"/>
        </body>"""

    blades = "\n".join(blade(i + 1, rx) for i, rx in enumerate(radials))

    return f"""\
    <!-- {n} | tower_height = {_f(H)} | pos ({_f(spec.x)}, {_f(spec.y)}) | yaw {_f(yaw_deg)} deg -->
    <body name="{n}_base" pos="{_f(spec.x)} {_f(spec.y)} {_f(spec.z)}" euler="0 0 {_f(yaw_deg)}">
      <geom name="{n}_tower_base" type="box" size="{_f(base_half)} {_f(base_half)} 0.1" pos="0 0 0" material="base_mat"/>
      <geom name="{n}_tower" type="cylinder" size="{_f(tower_radius)} {_f(H/2)}" pos="0 0 {_f(tower_z)}" material="tower_mat"/>
      <geom name="{n}_nacelle" type="box" size="{_f(nacelle_len/2)} {_f(tower_radius)} {_f(nacelle_h/2)}" pos="{_f(-nacelle_len/4)} 0 {_f(nacelle_z)}" material="nacelle_mat"/>

      <body name="{n}_hub" pos="{_f(hub_x)} 0 {_f(hub_z)}">
        <joint name="{n}_rotor" class="rotor"/>
        <geom name="{n}_hub_geom" type="cylinder" size="{_f(hub_r)} {_f(hub_hl)}" euler="0 90 0" pos="0 0 0" material="hub_mat"/>
        <geom name="{n}_spinner" type="ellipsoid" size="{_f(spin_a)} {_f(spin_b)} {_f(spin_b)}" pos="{_f(0.04*H)} 0 0" material="hub_mat"/>

{blades}
      </body>
    </body>"""


def build_mujoco_xml(specs: list[TurbineSpec], ground: float = 40.0) -> str:
    """Return a complete MuJoCo XML string for the whole farm."""
    bodies = "\n\n".join(_turbine_body(s) for s in specs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Auto-generated from main.py's WIND_FARM by wind_farm_export.py. Do not hand-edit. -->
<mujoco model="wind_turbines">

  <compiler eulerseq="xyz" angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <rgba haze="0.55 0.7 0.85 1"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <default class="rotor">
      <joint type="hinge" axis="1 0 0" limited="false" damping="0.25" frictionloss="0.02"/>
    </default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.95" rgb2="0.05 0.1 0.2" width="512" height="512"/>
    <texture name="ground_tex" type="2d" builtin="checker"
             rgb1="0.7 0.72 0.74" rgb2="0.55 0.57 0.6" width="300" height="300"/>
    <material name="ground_mat"  texture="ground_tex" texrepeat="10 10" reflectance="0.15"/>
    <material name="tower_mat"   rgba="0.93 0.93 0.95 1" specular="0.4" shininess="0.4"/>
    <material name="base_mat"    rgba="0.45 0.42 0.4  1" specular="0.1"/>
    <material name="nacelle_mat" rgba="0.85 0.86 0.88 1" specular="0.5" shininess="0.5"/>
    <material name="hub_mat"     rgba="0.8  0.82 0.85 1" specular="0.6" shininess="0.6"/>
    <material name="blade_mat"   rgba="0.96 0.96 0.97 1" specular="0.5" shininess="0.5"/>
    <material name="tip_mat"     rgba="0.85 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 30" dir="0 0 -1" directional="true"
           diffuse="0.6 0.6 0.6" specular="0.3 0.3 0.3"/>
    <geom name="ground" type="plane" size="{_f(ground)} {_f(ground)} 0.1" material="ground_mat" contype="1" conaffinity="1"/>

{bodies}

  </worldbody>
</mujoco>
"""


def export_wind_farm(specs: list[TurbineSpec] = None,
                     out_path: str = "wind_turbine_generated.xml") -> Path:
    """Write the farm to a MuJoCo XML file and return its path."""
    specs = specs if specs is not None else WIND_FARM
    xml = build_mujoco_xml(specs)
    p = Path(out_path)
    p.write_text(xml, encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export main.py's wind farm to a MuJoCo XML for wind_turbine_sim.py")
    ap.add_argument("--out", default="wind_turbine_generated.xml", help="output XML path")
    ap.add_argument("--launch", action="store_true", help="open wind_turbine_sim.py viewer after exporting")
    ap.add_argument("--wind", type=float, default=8.0, help="initial wind speed for the viewer (m/s)")
    ap.add_argument("--headless", type=float, default=None, help="run sim headless N seconds instead of viewer")
    args = ap.parse_args()

    path = export_wind_farm(WIND_FARM, args.out)
    print(f"Wrote {len(WIND_FARM)} turbines -> {path.resolve()}")

    if args.launch or args.headless is not None:
        cmd = [sys.executable, "wind_turbine_sim.py", "--model", str(path), "--wind", str(args.wind)]
        if args.headless is not None:
            cmd += ["--headless", str(args.headless)]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
