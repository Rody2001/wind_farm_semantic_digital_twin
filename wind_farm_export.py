"""wind_farm_export.py - export wind farms to one MuJoCo scene (user's uploaded version + repeatable --farm)."""
from __future__ import annotations
import argparse, math, subprocess, sys
from dataclasses import dataclass, replace
from pathlib import Path

from history_file import clear_history

PITCH_DEG = 18.0

R_TOWER_WIDTH    = 901  / 15797
R_NACELLE_LEN    = 1500 / 15797
R_NACELLE_HEIGHT = 678  / 15797
R_BLADE_LENGTH   = 8475 / 15797
R_BLADE_X        = 0.04 / 5
R_BLADE_Y        = 0.18 / 5

# How large one tile of the ground texture should be, in metres. Deriving the
# texture repeat from this keeps a tile the same real-world size whatever the
# farms' extent, so it stays a usable size reference next to a 131 m tower.
GROUND_TILE_M = 20.0        # for a photographic texture (--ground)
CHECKER_TILE_M = 20.0       # for the procedural checkerboard


@dataclass
class TurbineSpec:
    name: str; tower_height: float
    x: float=0.0; y: float=0.0; z: float=0.1; yaw: float=0.0; rotor_blade_length: float=0.0


# ===================================================================
# SINGLE SOURCE OF TRUTH
# ===================================================================
WIND_FARM_SINGLE: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=131.0, x=100, y=0,  z=0.1, rotor_blade_length=69),
]
WIND_FARM_East: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=131.0, x=100, y=0,  z=0.1, rotor_blade_length=69.125),
    TurbineSpec("2", tower_height=131.0, x=150, y=50,  z=0.1, rotor_blade_length=69.125),
    TurbineSpec("3", tower_height=131.0, x=200, y=100, z=0.1, rotor_blade_length=69.125),
    TurbineSpec("4", tower_height=131.0, x=250, y=150, z=0.1, rotor_blade_length=69.125),
    TurbineSpec("5", tower_height=131.0, x=300, y=200, z=0.1, rotor_blade_length=69.125),
    TurbineSpec("6", tower_height=131.0, x=350, y=250, z=0.1, rotor_blade_length=69.125),
]

WIND_FARM_West: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=81.0, x=-100, y=0, z=0.1, yaw=math.pi,),
    TurbineSpec("2", tower_height=81.0, x=-150, y=-50, z=0.1, yaw=math.pi,),
    TurbineSpec("3", tower_height=81.0, x=-200, y=-100, z=0.1, yaw=math.pi,),
    # TurbineSpec("4", tower_height=131.0, x=-250, y=-150, z=0.1, yaw=math.pi, rotor_blade_length=69.125),
    # TurbineSpec("5", tower_height=131.0, x=-300, y=-200, z=0.1, yaw=math.pi, rotor_blade_length=69.125),
    # TurbineSpec("6", tower_height=131.0, x=-350, y=-250, z=0.1, yaw=math.pi, rotor_blade_length=69.125),
]
WIND_FARM_North: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=111.0, x=0, y=100, z=0.1, yaw=math.pi/2,),
    TurbineSpec("2", tower_height=111.0, x=-50, y=150, z=0.1, yaw=math.pi/2,),
    # TurbineSpec("3", tower_height=131.0, x=-100, y=200, z=0.1, yaw=math.pi/2, rotor_blade_length=69.125),
    # TurbineSpec("4", tower_height=131.0, x=-150, y=250, z=0.1, yaw=math.pi/2, rotor_blade_length=69.125),
    # TurbineSpec("5", tower_height=131.0, x=-200, y=300, z=0.1, yaw=math.pi/2, rotor_blade_length=69.125),
    # TurbineSpec("6", tower_height=131.0, x=-250, y=350, z=0.1, yaw=math.pi/2, rotor_blade_length=69.125),
]
WIND_FARM_South: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=160.0, x=0, y=-100, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
    TurbineSpec("2", tower_height=160.0, x=50, y=-150, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
    TurbineSpec("3", tower_height=160.0, x=100, y=-200, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
    TurbineSpec("4", tower_height=160.0, x=150, y=-250, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
    TurbineSpec("5", tower_height=160.0, x=200, y=-300, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
    # TurbineSpec("6", tower_height=131.0, x=250, y=-350, z=0.1, yaw=3*math.pi/2, rotor_blade_length=69.125),
]

WIND_FARM_Big: list[TurbineSpec] = [
    TurbineSpec("1", tower_height=300.0, x=0, y=0,  z=0.1, yaw=0.0),
]

# Every farm that should be exported / started together.
ALL_FARMS: dict[str, list[TurbineSpec]] = {
    # "Farm_East": WIND_FARM_East,
    # "Farm_North": WIND_FARM_North,
    # "Farm_West": WIND_FARM_West,
    # "Farm_South": WIND_FARM_South,
    # "Farm_Big": WIND_FARM_Big,
    "Farm_Single": WIND_FARM_SINGLE,
}


def combined_specs(farms: dict[str, list[TurbineSpec]] = None) -> list[TurbineSpec]:
    """Flatten all farms into one list, prefixing names with the farm label."""
    farms = farms if farms is not None else ALL_FARMS
    out: list[TurbineSpec] = []
    for label, specs in farms.items():
        for s in specs:
            out.append(replace(s, name=f"{label}_{s.name}"))
    return out


# ------------------------------------------------------------------- #
def _f(v: float) -> str:
    """Format a float for XML."""
    return f"{v:.6g}"


class MeshLibrary:
    """Collects the <mesh> assets a scene needs, one per part and scale.

    Turbines of the same height share a mesh asset; a different height needs a
    different scale and therefore its own asset, even though both point at the
    same OBJ file.
    """

    def __init__(self, directory: str):
        self.dir = directory.rstrip("/")
        self._names: dict = {}
        self._order: list = []

    def ref(self, part: str, sx: float, sy: float, sz: float) -> str:
        key = (part, round(sx, 6), round(sy, 6), round(sz, 6))
        if key not in self._names:
            name = f"{part}_{len(self._order)}"
            self._names[key] = name
            self._order.append((name, part, sx, sy, sz))
        return self._names[key]

    def xml(self) -> str:
        return "\n    ".join(
            f'<mesh name="{n}" file="{self.dir}/{p}.obj" '
            f'scale="{_f(sx)} {_f(sy)} {_f(sz)}"/>'
            for n, p, sx, sy, sz in self._order)


def _turbine_body(spec: TurbineSpec, meshes: "MeshLibrary" = None) -> str:
    """Build a MuJoCo XML body for a single turbine.

    With `meshes`, every part is a scaled instance of a generated OBJ; without
    it, the original primitives are used. The joints, the sites, and every geom
    name are identical either way, so the drivers cannot tell the difference.
    """
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
    hub_x     = nacelle_len / 4.0 + tower_radius   # now relative to the nacelle body, not the tower

    hub_r     = 0.03 * H
    hub_hl    = 0.02 * H
    spin_a, spin_b = 0.05 * H, 0.028 * H

    # The hub body is placed hub_x ahead of the nacelle body, which puts it
    # tower_radius beyond the nacelle's front face -- the rotor overhang a real
    # turbine has. In the hub's own frame that face therefore lies at
    # x = -tower_radius. Anything drawn on the hub has to start behind that
    # point, or the spinner floats free of the nacelle.
    hub_back  = -1.3 * tower_radius
    nose_x    = 0.09 * H

    blade_z   = L / 2.0 + nacelle_h / 2.0
    tip_half  = 0.03 * L
    tip_z     = blade_z + L / 2.0 - tip_half
    aero_z    = blade_z + 0.055 * L
    aero_r    = 0.006 * H
    radials   = (-120.0, 120.0, 0.0)

    # blade meshes span z in [0, 1] from the root, so they hang from the hub
    blade_root_z = nacelle_h / 2.0

    def blade(i: int, rx: float) -> str:
        """Build a MuJoCo XML body for a single blade."""
        b = f"{n}_blade{i}"
        if meshes is not None:
            m_blade = meshes.ref("blade", blade_x_half * 2, blade_y_half * 2, L)
            m_tip = meshes.ref("blade_tip", blade_x_half * 2, blade_y_half * 2, L)
            geoms = (
                f'<geom name="{b}_geom" type="mesh" mesh="{m_blade}" '
                f'pos="0 0 {_f(blade_root_z)}" material="blade_mat"/>\n'
                f'          <geom name="{b}_tip" type="mesh" mesh="{m_tip}" '
                f'pos="0 0 {_f(blade_root_z)}" material="tip_mat"/>')
        else:
            geoms = (
                f'<geom name="{b}_geom" type="box" '
                f'size="{_f(blade_x_half)} {_f(blade_y_half)} {_f(L/2)}" '
                f'pos="0 0 {_f(blade_z)}" material="blade_mat"/>\n'
                f'          <geom name="{b}_tip" type="box" '
                f'size="{_f(blade_x_half)} {_f(blade_y_half)} {_f(tip_half)}" '
                f'pos="0 0 {_f(tip_z)}" material="tip_mat"/>')
        return f"""\
        <body name="{b}" euler="{_f(rx)} 0 {_f(-PITCH_DEG)}">
          {geoms}
          <site name="{b}_aero" pos="0 0 {_f(aero_z)}" size="{_f(aero_r)}"/>
        </body>"""

    blades = "\n".join(blade(i + 1, rx) for i, rx in enumerate(radials))

    if meshes is not None:
        base_geom = (f'<geom name="{n}_tower_base" type="mesh" '
                     f'mesh="{meshes.ref("base", base_half, base_half, 0.2)}" '
                     f'pos="0 0 0" material="base_mat"/>')
        tower_geom = (f'<geom name="{n}_tower" type="mesh" '
                      f'mesh="{meshes.ref("tower", tower_radius, tower_radius, H)}" '
                      f'pos="0 0 0.1" material="tower_mat"/>')
        nacelle_geom = (f'<geom name="{n}_nacelle_geom" type="mesh" '
                        f'mesh="{meshes.ref("nacelle", nacelle_len, tower_width, nacelle_h)}" '
                        f'pos="{_f(-nacelle_len/4)} 0 0" material="nacelle_mat"/>')
        hub_geoms = (f'<geom name="{n}_spinner" type="mesh" '
                     f'mesh="{meshes.ref("spinner", nose_x - hub_back, hub_r, hub_r)}" '
                     f'pos="{_f(hub_back)} 0 0" material="hub_mat"/>')
    else:
        base_geom = (f'<geom name="{n}_tower_base" type="box" '
                     f'size="{_f(base_half)} {_f(base_half)} 0.1" pos="0 0 0" material="base_mat"/>')
        tower_geom = (f'<geom name="{n}_tower" type="cylinder" '
                      f'size="{_f(tower_radius)} {_f(H/2)}" pos="0 0 {_f(tower_z)}" material="tower_mat"/>')
        nacelle_geom = (f'<geom name="{n}_nacelle_geom" type="box" '
                        f'size="{_f(nacelle_len/2)} {_f(tower_radius)} {_f(nacelle_h/2)}" '
                        f'pos="{_f(-nacelle_len/4)} 0 0" material="nacelle_mat"/>')
        # stretch the hub cylinder back to the nacelle instead of centring it
        hub_half = (hub_hl - hub_back) / 2.0
        hub_cx = (hub_hl + hub_back) / 2.0
        hub_geoms = (f'<geom name="{n}_hub_geom" type="cylinder" '
                     f'size="{_f(hub_r)} {_f(hub_half)}" euler="0 90 0" '
                     f'pos="{_f(hub_cx)} 0 0" material="hub_mat"/>\n'
                     f'          <geom name="{n}_spinner" type="ellipsoid" '
                     f'size="{_f(spin_a)} {_f(spin_b)} {_f(spin_b)}" '
                     f'pos="{_f(0.04*H)} 0 0" material="hub_mat"/>')

    return f"""\
    <!-- {n} | tower_height = {_f(H)} | pos ({_f(spec.x)}, {_f(spec.y)}) | initial yaw {_f(yaw_deg)} deg -->
    <body name="{n}_base" pos="{_f(spec.x)} {_f(spec.y)} {_f(spec.z)}" euler="0 0 {_f(yaw_deg)}">
      {base_geom}
      {tower_geom}

      <body name="{n}_nacelle" pos="0 0 {_f(nacelle_z)}">
        <joint name="{n}_yaw" class="yaw"/>
        {nacelle_geom}

        <body name="{n}_hub" pos="{_f(hub_x)} 0 0">
          <joint name="{n}_rotor" class="rotor"/>
          {hub_geoms}

{blades}
        </body>
      </body>
    </body>"""


def _auto_ground(specs: list[TurbineSpec]) -> float:
    """Estimate the ground height based on the turbine positions."""
    if not specs:
        return 40.0
    reach = max(max(abs(s.x), abs(s.y)) + 1.5 * s.tower_height for s in specs)
    return reach + 50.0


def _sky_texture(sky_file: str = None) -> str:
    """The skybox asset: a photographic cube map if one was given, else a gradient.

    A cube-map cross PNG is what make_skybox.py produces from an equirectangular
    panorama; MuJoCo cannot read equirectangular images itself.
    """
    if sky_file:
        return (f'<texture name="sky" type="skybox" file="{sky_file}"\n'
                f'             gridsize="3 4" gridlayout=".U..LFRB.D.."/>')
    return ('<texture name="sky" type="skybox" builtin="gradient"\n'
            '             rgb1="0.26 0.42 0.70" rgb2="0.80 0.87 0.94"\n'
            '             width="512" height="3072"/>')


def _ground_texture(ground_file: str = None) -> tuple[str, float]:
    """The ground asset and how many times it repeats across the plane.

    The plane spans 2 * ground metres, so the repeat count that gives a tile of
    TILE metres is (2 * ground) / TILE. Returned as a factor to multiply by the
    ground size, since the caller knows that number.
    """
    if ground_file:
        return (f'<texture name="ground_tex" type="2d" file="{ground_file}"/>',
                2.0 / GROUND_TILE_M)
    return ('<texture name="ground_tex" type="2d" builtin="checker"\n'
            '             rgb1="0.29 0.40 0.25" rgb2="0.24 0.35 0.21"\n'
            '             mark="edge" markrgb="0.33 0.45 0.29"\n'
            '             width="512" height="512"/>',
            2.0 / (2.0 * CHECKER_TILE_M))   # a checker texture is 2x2 squares


def build_mujoco_xml(specs: list[TurbineSpec], ground: float = None,
                     sky_file: str = None, ground_file: str = None,
                     mesh_dir: str = None) -> str:
    """Build a MuJoCo XML file for a wind farm."""
    if ground is None:
        ground = _auto_ground(specs)

    meshes = MeshLibrary(mesh_dir) if mesh_dir else None
    bodies = "\n\n".join(_turbine_body(s, meshes) for s in specs)
    mesh_assets = ("\n    " + meshes.xml()) if meshes else ""

    sky_tex = _sky_texture(sky_file)
    ground_tex, repeat_factor = _ground_texture(ground_file)
    tiles = ground * repeat_factor

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Auto-generated by wind_farm_export.py ({len(specs)} turbines, all farms). Do not hand-edit. -->
<mujoco model="wind_turbines">

  <compiler eulerseq="xyz" angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.42 0.42 0.42" specular="0.1 0.1 0.1"/>
    <map haze="0.3"/>
    <rgba haze="0.82 0.87 0.92 1"/>
    <quality shadowsize="8192" offsamples="8"/>
  </visual>

  <default>
    <default class="rotor">
      <joint type="hinge" axis="1 0 0" limited="false" damping="0.25" frictionloss="0.02"/>
    </default>
    <default class="yaw">
      <joint type="hinge" axis="0 0 1" limited="false" damping="0.25" frictionloss="0.02"/>
    </default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <asset>{mesh_assets}
    {sky_tex}
    {ground_tex}
    <material name="ground_mat"  texture="ground_tex" texrepeat="{_f(tiles)} {_f(tiles)}"
              reflectance="0.04" specular="0.1" shininess="0.1"/>
    <material name="tower_mat"   rgba="0.93 0.93 0.95 1" specular="0.4" shininess="0.4"/>
    <material name="base_mat"    rgba="0.45 0.42 0.4  1" specular="0.1"/>
    <material name="nacelle_mat" rgba="0.85 0.86 0.88 1" specular="0.5" shininess="0.5"/>
    <material name="hub_mat"     rgba="0.8  0.82 0.85 1" specular="0.6" shininess="0.6"/>
    <material name="blade_mat"   rgba="0.96 0.96 0.97 1" specular="0.5" shininess="0.5"/>
    <material name="tip_mat"     rgba="0.85 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="{_f(ground*0.5)} {_f(-ground*0.5)} {_f(ground)}"
           dir="-0.4 0.4 -1" directional="true" castshadow="true"
           diffuse="0.85 0.82 0.75" specular="0.25 0.25 0.25"/>
    <geom name="ground" type="plane" size="{_f(ground)} {_f(ground)} 0.1" material="ground_mat" contype="1" conaffinity="1"/>

{bodies}

  </worldbody>
</mujoco>
"""


def export_wind_farm(specs: list[TurbineSpec] = None,
                     out_path: str = "wind_turbine_generated.xml",
                     sky_file: str = None, ground_file: str = None,
                     mesh_dir: str = None) -> Path:
    """Export a wind farm to a MuJoCo XML file."""
    specs = specs if specs is not None else combined_specs()
    Path(out_path).write_text(
        build_mujoco_xml(specs, sky_file=sky_file, ground_file=ground_file,
                         mesh_dir=mesh_dir),
        encoding="utf-8")
    return Path(out_path)

def labels_for_token(token):
    """Resolve one --farm token to matching ALL_FARMS labels."""
    t=token.strip().lower()
    if t=="all": return list(ALL_FARMS)
    exact=[k for k in ALL_FARMS if k.lower()==t]
    if exact: return exact
    if not t.startswith("farm_"):
        cand=[k for k in ALL_FARMS if k.lower()==f"farm_{t}"]
        if cand: return cand
    return [k for k in ALL_FARMS if t in k.lower()]

def resolve_farms(tokens):
    """Union several --farm tokens into one ordered {label: specs} dict."""
    chosen=set()
    for tok in tokens:
        labels=labels_for_token(tok)
        if not labels: raise SystemExit(f"--farm '{tok}' matched no farms. Valid: {list(ALL_FARMS)}")
        chosen.update(labels)
    return {k:ALL_FARMS[k] for k in ALL_FARMS if k in chosen}

def main() -> None:
    """Export wind farm(s) to a MuJoCo XML and optionally launch the sim."""
    ap = argparse.ArgumentParser(description="Export wind farm(s) to a MuJoCo XML for wind_turbine_sim.py")
    ap.add_argument("--out", default="wind_turbine_generated.xml", help="output XML path")
    ap.add_argument("--farm", action="append", default=None,
                    help="farm(s) to include; REPEATABLE. e.g. --farm Farm_South --farm Farm_West "
                         "(or short: --farm south --farm west). Default: all.")
    ap.add_argument("--launch", action="store_true", help="open the viewer after exporting")
    ap.add_argument("--wind", type=float, default=8.0, help="initial wind speed for the viewer (m/s)")
    ap.add_argument("--gridLimit", type=float, default=None,
                    help="grid UPPER limit in MW; only run enough turbines to stay under it")
    ap.add_argument("--direction", type=float, default=0.0,
                    help="wind comes FROM this compass bearing in degrees (0=N, 90=E, 180=S, "
                         "270=W); change live in the viewer with 'd' then Up/Down arrows")
    ap.add_argument("--yaw-rate", type=float, default=None,
                    help="nacelle yaw slew rate in deg/s while tracking the wind (default: "
                         "the viewer's own default, currently 10 deg/s)")
    ap.add_argument("--rotor-accel", type=float, default=None,
                    help="max RPM change per second (rotor inertia); default: the viewer's "
                         "own default, currently 1.0 RPM/s")
    ap.add_argument("--temp", type=float, default=None,
                    help="initial air temperature in deg C, sets rho via the ideal gas law "
                         "(default: the viewer's own default, currently 15 degC -> rho=1.225)")
    ap.add_argument("--all-spin", action="store_true",
                    help="disable directional gating: every turbine spins regardless of facing")
    ap.add_argument("--publish", action="store_true", help="publish rpm/power/energy to ROS 2 topics")
    ap.add_argument("--ui", action="store_true",
                    help="open the browser control panel: sliders for wind speed, temperature "
                         "and grid limit, compass for wind direction, live while the sim runs")
    ap.add_argument("--ui-port", type=int, default=8080,
                    help="port for the --ui control panel (default 8080)")
    ap.add_argument("--sky", default=None,
                    help="skybox cube-map cross PNG, as produced by make_skybox.py from an "
                         "equirectangular panorama (default: a procedural gradient sky)")
    ap.add_argument("--ground", default=None,
                    help="ground colour texture PNG, e.g. a CC0 grass texture from ambientCG "
                         "(default: a procedural checkerboard)")
    ap.add_argument("--meshes", default=None,
                    help="directory of turbine part meshes from make_turbine_meshes.py "
                         "(default: the boxes, cylinders and ellipsoids)")
    ap.add_argument("--time", type=float, default=None,
                    help="run N seconds then auto-close, logging history at 1 Hz")
    ap.add_argument("--history-file", default="history.jsonl",
                    help="path for the 1 Hz history log (used with --time)")
    ap.add_argument("--headless", type=float, default=None, help="run sim headless N seconds instead of viewer")
    args = ap.parse_args()

    for label, value in (("--sky", args.sky), ("--ground", args.ground)):
        if value is not None and not Path(value).is_file():
            raise SystemExit(f"{label}: no such file: {value}")
    if args.meshes is not None:
        missing = [f"{p}.obj" for p in ("base", "tower", "nacelle", "spinner",
                                        "blade", "blade_tip")
                   if not (Path(args.meshes) / f"{p}.obj").is_file()]
        if missing:
            raise SystemExit(f"--meshes {args.meshes}: missing {', '.join(missing)}. "
                             f"Run: python make_turbine_meshes.py -o {args.meshes}")

    selected = resolve_farms(args.farm or ["all"])
    specs = combined_specs(selected)

    path = export_wind_farm(specs, args.out, args.sky, args.ground, args.meshes)
    counts = ", ".join(f"{k}: {len(v)}" for k, v in selected.items())
    print(f"Wrote {len(specs)} turbines ({counts}) -> {path.resolve()}")
    print(f"  sky:    {args.sky or 'procedural gradient'}")
    print(f"  ground: {args.ground or 'procedural checkerboard'}")
    print(f"  parts:  {args.meshes + '/*.obj' if args.meshes else 'primitives'}")

    if args.launch or args.headless is not None:
        cmd = [sys.executable, "wind_turbine_sim.py", "--model", str(path),
               "--wind", str(args.wind), "--direction", str(args.direction)]
        if args.gridLimit is not None:
            cmd += ["--gridLimit", str(args.gridLimit)]
        if args.yaw_rate is not None:
            cmd += ["--yaw-rate", str(args.yaw_rate)]
        if args.rotor_accel is not None:
            cmd += ["--rotor-accel", str(args.rotor_accel)]
        if args.temp is not None:
            cmd += ["--temp", str(args.temp)]
        if args.all_spin:
            cmd += ["--all-spin"]
        if args.publish:
            cmd += ["--publish"]
        if args.ui:
            cmd += ["--ui", "--ui-port", str(args.ui_port)]
        if args.time is not None:
            clear_history()
            cmd += ["--time", str(args.time), "--history-file", args.history_file]
        if args.headless is not None:
            cmd += ["--headless", str(args.headless)]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()