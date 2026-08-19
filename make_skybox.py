#!/usr/bin/env python3
"""
make_skybox.py
===================================================================
Turn an equirectangular sky panorama into the cube-map cross image that
MuJoCo's skybox wants.

MuJoCo loads a skybox either from six separate images or from one composite
image described by ``gridsize`` and ``gridlayout``. It does not understand
equirectangular panoramas, which is the format every free HDRI site publishes.
This script does the conversion.

    python make_skybox.py sky.jpg -o assets/skybox.png --size 1024

The result is a 4N x 3N PNG in the layout MuJoCo documents:

        .  U  .  .
        L  F  R  B
        .  D  .  .

referenced from MJCF as::

    <texture name="sky" type="skybox" file="assets/skybox.png"
             gridsize="3 4" gridlayout=".U..LFRB.D.."/>

Where to get a panorama (both CC0, no attribution required):

    https://polyhaven.com/hdris  -- pick one, open the download menu next to
                                    the button and choose "Tonemapped JPG".
                                    The "pure sky" ones have no scenery on the
                                    horizon, which suits a wind farm.
    https://ambientcg.com/       -- has HDRIs as well as ground textures.

Only the standard library plus numpy and Pillow are needed:

    pip install numpy pillow
===================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# For each cube face: the direction it looks along, the direction image x runs
# in, and the direction image y runs in. All in MuJoCo world axes (z is up).
# Chosen so the faces join seamlessly in the cross layout above: U sits on top
# of F, D below it, and L F R B run continuously across the middle row.
FACES = {
    "F": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "R": ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
    "B": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "L": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "U": ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
    "D": ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
}

# (row, column) of each face in the 3x4 grid, matching ".U..LFRB.D.."
GRID = {"U": (0, 1), "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3), "D": (2, 1)}


def render_face(pano: np.ndarray, face: str, size: int, yaw_deg: float) -> np.ndarray:
    """Sample one cube face out of an equirectangular panorama."""
    forward, right, down = (np.array(v, dtype=np.float64) for v in FACES[face])

    # rotate the whole cube about the vertical axis, to place the sun where you want it
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    forward, right, down = rot @ forward, rot @ right, rot @ down

    # a, b run from -1 to 1 across the face, sampled at pixel centres
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    a, b = np.meshgrid(t, t)
    dirs = (forward[None, None, :]
            + a[..., None] * right[None, None, :]
            + b[..., None] * down[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)

    h, w = pano.shape[:2]
    lon = np.arctan2(dirs[..., 1], dirs[..., 0])       # -pi .. pi
    lat = np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0))  # -pi/2 .. pi/2 (up positive)

    u = (lon + np.pi) / (2.0 * np.pi) * w
    v = (np.pi / 2.0 - lat) / np.pi * h                # row 0 of the panorama is the zenith

    ui = np.clip(u.astype(np.int64), 0, w - 1)
    vi = np.clip(v.astype(np.int64), 0, h - 1)
    return pano[vi, ui]


def build_skybox(pano: np.ndarray, size: int, yaw_deg: float) -> np.ndarray:
    """Assemble the six faces into the 3x4 cross MuJoCo expects."""
    sheet = np.zeros((size * 3, size * 4, 3), dtype=pano.dtype)
    for face, (row, col) in GRID.items():
        sheet[row * size:(row + 1) * size, col * size:(col + 1) * size] = \
            render_face(pano, face, size, yaw_deg)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an equirectangular sky panorama into a MuJoCo skybox PNG")
    ap.add_argument("panorama", help="equirectangular image (the Tonemapped JPG from Poly Haven)")
    ap.add_argument("-o", "--out", default="skybox.png", help="output PNG (default skybox.png)")
    ap.add_argument("--size", type=int, default=1024,
                    help="pixels per cube face (default 1024; the file is 4x3 of these)")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="rotate the sky about the vertical axis in degrees, to move the "
                         "sun relative to the farm (default 0)")
    args = ap.parse_args()

    src = Path(args.panorama)
    pano = np.asarray(Image.open(src).convert("RGB"))
    print(f"panorama: {src.name}  {pano.shape[1]}x{pano.shape[0]}")

    if pano.shape[1] != 2 * pano.shape[0]:
        print("  note: this is not a 2:1 image, so it may not be equirectangular; "
              "the result will still be produced but may look distorted")

    sheet = build_skybox(pano, args.size, args.yaw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"wrote {out}  {sheet.shape[1]}x{sheet.shape[0]}")
    print('\nreference it from MJCF with:\n'
          f'  <texture name="sky" type="skybox" file="{out}"\n'
          '           gridsize="3 4" gridlayout=".U..LFRB.D.."/>')


if __name__ == "__main__":
    main()
