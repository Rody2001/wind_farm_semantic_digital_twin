"""
wind_farm_export.py
===================================================================
ONE-BUTTON EXPORT:  turn the wind farms defined here into a single
MuJoCo XML scene that `wind_turbine_sim.py` can drive directly.

All farms listed in ALL_FARMS are exported into ONE scene and started
together. Turbine names are prefixed with their farm label so the two
farms (which reuse turbine1..turbine6) don't collide.

Run (the button):
    python wind_farm_export.py                 # export ALL farms -> wind_turbine_generated.xml
    python wind_farm_export.py --launch        # export ALL farms, then open the viewer
    python wind_farm_export.py --farm A        # only farm A
    python wind_farm_export.py --launch --wind 12
===================================================================
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PITCH_DEG = 18.0

R_TOWER_WIDTH    = 901  / 15797
R_NACELLE_LEN    = 1500 / 15797
R_NACELLE_HEIGHT = 678  / 15797
R_BLADE_LENGTH   = 8475 / 15797
R_BLADE_X        = 0.04 / 5
R_BLADE_Y        = 0.18 / 5


@dataclass
class TurbineSpec:
    """One turbine in the farm. Mirrors a WindTurbine.create_* call."""
    name: str
    tower_height: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.1
    yaw: float = 0.0
    rotor_blade_length: float = 0.0


# ===================================================================
# SINGLE SOURCE OF TRUTH
# ===================================================================
WIND_FARM_A_small: list[TurbineSpec] = [
    TurbineSpec("turbine_A_small_1", tower_height=111.0, x=0,   z=0.1),
    TurbineSpec("turbine_A_small_2", tower_height=111.0, x=50,  z=0.1),
    TurbineSpec("turbine_A_small_3", tower_height=111.0, x=100, z=0.1),
    TurbineSpec("turbine_A_small_4", tower_height=111.0, x=150, z=0.1),
    TurbineSpec("turbine_A_small_5", tower_height=111.0, x=200, z=0.1),
    TurbineSpec("turbine_A_small_6", tower_height=111.0, x=250, z=0.1),
]
WIND_FARM_A_big: list[TurbineSpec] = [
    TurbineSpec("turbine_A_big_1", tower_height=160.0, x=0,  y=150, z=0.1),
    TurbineSpec("turbine_A_big_2", tower_height=160.0, x=50, y=150, z=0.1),
    TurbineSpec("turbine_A_big_3", tower_height=160.0, x=100,y=150, z=0.1),
    TurbineSpec("turbine_A_big_4", tower_height=160.0, x=150,y=150, z=0.1),
    TurbineSpec("turbine_A_big_5", tower_height=160.0, x=200,y=150, z=0.1),
    TurbineSpec("turbine_A_big_6", tower_height=160.0, x=250,y=150, z=0.1),
]
WIND_FARM_B_small: list[TurbineSpec] = [
    TurbineSpec("turbine_B_small_1", tower_height=111.0, x=-100, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_small_2", tower_height=111.0, x=-150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_small_3", tower_height=111.0, x=-200, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_small_4", tower_height=111.0, x=-250, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_small_5", tower_height=111.0, x=-300, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_small_6", tower_height=111.0, x=-350, z=0.1, yaw=math.pi),
]
WIND_FARM_B_big: list[TurbineSpec] = [
    TurbineSpec("turbine_B_big_1", tower_height=160.0, x=-100, y=150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_big_2", tower_height=160.0, x=-150, y=150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_big_3", tower_height=160.0, x=-200, y=150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_big_4", tower_height=160.0, x=-250, y=150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_big_5", tower_height=160.0, x=-300, y=150, z=0.1, yaw=math.pi),
    TurbineSpec("turbine_B_big_6", tower_height=160.0, x=-350, y=150, z=0.1, yaw=math.pi),
]
WIND_FARM_C_small: list[TurbineSpec] = [
    TurbineSpec("turbine_C_small_1", tower_height=111.0, y=250, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_small_2", tower_height=111.0, y=300, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_small_3", tower_height=111.0, y=350, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_small_4", tower_height=111.0, y=400, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_small_5", tower_height=111.0, y=450, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_small_6", tower_height=111.0, y=500, z=0.1, yaw=math.pi/2),
]
WIND_FARM_C_big: list[TurbineSpec] = [
    TurbineSpec("turbine_C_big_1", tower_height=160.0, x=-150, y=250, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_big_2", tower_height=160.0, x=-150, y=300, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_big_3", tower_height=160.0, x=-150, y=350, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_big_4", tower_height=160.0, x=-150, y=400, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_big_5", tower_height=160.0, x=-150, y=450, z=0.1, yaw=math.pi/2),
    TurbineSpec("turbine_C_big_6", tower_height=160.0, x=-150, y=500, z=0.1, yaw=math.pi/2),
]
WIND_FARM_D_small: list[TurbineSpec] = [
    TurbineSpec("turbine_D_small_1", tower_height=111.0, x=-150, y=-100, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_small_2", tower_height=111.0, x=-150, y=-150, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_small_3", tower_height=111.0, x=-150, y=-200, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_small_4", tower_height=111.0, x=-150, y=-250, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_small_5", tower_height=111.0, x=-150, y=-300, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_small_6", tower_height=111.0, x=-150, y=-350, z=0.1, yaw=3*math.pi/2),
]
WIND_FARM_D_big: list[TurbineSpec] = [
    TurbineSpec("turbine_D_big_1", tower_height=160.0, y=-100, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_big_2", tower_height=160.0, y=-150, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_big_3", tower_height=160.0, y=-200, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_big_4", tower_height=160.0, y=-250, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_big_5", tower_height=160.0, y=-300, z=0.1, yaw=3*math.pi/2),
    TurbineSpec("turbine_D_big_6", tower_height=160.0, y=-350, z=0.1, yaw=3*math.pi/2),
]

# Every farm that should be exported / started together.
ALL_FARMS: dict[str, list[TurbineSpec]] = {
    "A_small": WIND_FARM_A_small,
    "A_big": WIND_FARM_A_big,
    "B_small": WIND_FARM_B_small,
    "B_big": WIND_FARM_B_big,
    "C_small": WIND_FARM_C_small,
    "C_big": WIND_FARM_C_big,
    "D_small": WIND_FARM_D_small,
    "D_big": WIND_FARM_D_big,
}


def combined_specs(farms: dict[str, list[TurbineSpec]] = None) -> list[TurbineSpec]:
    """Flatten all farms into one list, prefixing names with the farm label
    so they stay unique inside a single MuJoCo scene."""
    farms = farms if farms is not None else ALL_FARMS
    out: list[TurbineSpec] = []
    for label, specs in farms.items():
        for s in specs:
            out.append(replace(s, name=f"{label}_{s.name}"))
    return out


# ------------------------------------------------------------------- #
def _f(v: float) -> str:
    return f"{v:.6g}"


def _turbine_body(spec: TurbineSpec) -> str:
    H = spec.tower_height
    n = spec.name

    tower_width   = H * R_TOWER_WIDTH
    tower_radius  = tower_width / 2.0
    nacelle_len   = H * R_NACELLE_LEN
    nacelle_h     = H * R_NACELLE_HEIGHT
    blade_x_half  = (H * R_BLADE_X) / 2.0
    blade_y_half  = (H * R_BLADE_Y) / 2.0
    L = spec.rotor_blade_length or (H * R_BLADE_LENGTH)

    base_half     = H / 6.0
    yaw_deg       = math.degrees(spec.yaw)

    tower_z   = H / 2.0 + 0.1
    nacelle_z = H + 0.1 + nacelle_h / 2.0
    hub_z     = nacelle_z
    hub_x     = nacelle_len / 4.0 + tower_radius

    hub_r     = 0.03 * H
    hub_hl    = 0.02 * H
    spin_a, spin_b = 0.05 * H, 0.028 * H

    blade_z   = L / 2.0 + nacelle_h / 2.0
    tip_half  = 0.03 * L
    tip_z     = blade_z + L / 2.0 - tip_half
    aero_z    = blade_z + 0.055 * L
    aero_r    = 0.006 * H
    radials   = (-120.0, 120.0, 0.0)

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


def _auto_ground(specs: list[TurbineSpec]) -> float:
    """Ground half-size big enough to hold every turbine plus a margin."""
    if not specs:
        return 40.0
    reach = max(max(abs(s.x), abs(s.y)) + 1.5 * s.tower_height for s in specs)
    return reach + 50.0


def build_mujoco_xml(specs: list[TurbineSpec], ground: float = None) -> str:
    """Return a complete MuJoCo XML string for all the given turbines."""
    if ground is None:
        ground = _auto_ground(specs)
    bodies = "\n\n".join(_turbine_body(s) for s in specs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Auto-generated by wind_farm_export.py ({len(specs)} turbines, all farms). Do not hand-edit. -->
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
    <material name="ground_mat"  texture="ground_tex" texrepeat="40 40" reflectance="0.15"/>
    <material name="tower_mat"   rgba="0.93 0.93 0.95 1" specular="0.4" shininess="0.4"/>
    <material name="base_mat"    rgba="0.45 0.42 0.4  1" specular="0.1"/>
    <material name="nacelle_mat" rgba="0.85 0.86 0.88 1" specular="0.5" shininess="0.5"/>
    <material name="hub_mat"     rgba="0.8  0.82 0.85 1" specular="0.6" shininess="0.6"/>
    <material name="blade_mat"   rgba="0.96 0.96 0.97 1" specular="0.5" shininess="0.5"/>
    <material name="tip_mat"     rgba="0.85 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 {_f(ground)}" dir="0 0 -1" directional="true"
           diffuse="0.6 0.6 0.6" specular="0.3 0.3 0.3"/>
    <geom name="ground" type="plane" size="{_f(ground)} {_f(ground)} 0.1" material="ground_mat" contype="1" conaffinity="1"/>

{bodies}

  </worldbody>
</mujoco>
"""


def export_wind_farm(specs: list[TurbineSpec] = None,
                     out_path: str = "wind_turbine_generated.xml") -> Path:
    """Write the farm(s) to a MuJoCo XML file and return its path.
    Defaults to ALL farms combined."""
    specs = specs if specs is not None else combined_specs()
    xml = build_mujoco_xml(specs)
    p = Path(out_path)
    p.write_text(xml, encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export wind farm(s) to a MuJoCo XML for wind_turbine_sim.py")
    ap.add_argument("--out", default="wind_turbine_generated.xml", help="output XML path")
    ap.add_argument("--farm", default="all",
                    help="which farm to export: 'all' (default) or a label like A or B")
    ap.add_argument("--launch", action="store_true", help="open the viewer after exporting")
    ap.add_argument("--wind", type=float, default=8.0, help="initial wind speed for the viewer (m/s)")
    ap.add_argument("--needed", type=float, default=None,
                    help="required power output in MW; only enough turbines spin to meet it")
    ap.add_argument("--publish", action="store_true", help="publish rpm/power/energy to ROS 2 topics")
    ap.add_argument("--headless", type=float, default=None, help="run sim headless N seconds instead of viewer")
    args = ap.parse_args()

    if args.farm.lower() == "all":
        specs = combined_specs()
    elif args.farm in ALL_FARMS:
        specs = combined_specs({args.farm: ALL_FARMS[args.farm]})
    else:
        raise SystemExit(f"--farm must be 'all' or one of {list(ALL_FARMS)}")

    path = export_wind_farm(specs, args.out)
    counts = ", ".join(f"farm {k}: {len(v)}" for k, v in ALL_FARMS.items()) \
        if args.farm.lower() == "all" else f"farm {args.farm}"
    print(f"Wrote {len(specs)} turbines ({counts}) -> {path.resolve()}")

    if args.launch or args.headless is not None:
        cmd = [sys.executable, "wind_turbine_sim.py", "--model", str(path), "--wind", str(args.wind)]
        if args.needed is not None:
            cmd += ["--needed", str(args.needed)]
        if args.publish:
            cmd += ["--publish"]
        if args.headless is not None:
            cmd += ["--headless", str(args.headless)]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()