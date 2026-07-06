"""
wind_state_file.py
===================================================================
Tiny file-based IPC so the MuJoCo viewer (wind_turbine_sim.py) and the
semantic-digital-twin world (main1.py) can share one "current wind" value
across two separate Python processes, without needing ROS topics or any
shared memory.

Whichever process changes the wind calls write_wind_state(...); any other
process (or the same one) calls read_wind_state() to get the latest value.
Writes are atomic (write to a temp file, then os.replace) so a reader never
sees a half-written file.
===================================================================
"""

import json
import os
import tempfile
from typing import Tuple

# Shared across processes/users on the same machine by default. Override via
# the WIND_FARM_STATE_PATH env var if you need per-user/per-run isolation
# (e.g. running two independent farm sims side by side).
DEFAULT_PATH = os.environ.get(
    "WIND_FARM_STATE_PATH",
    os.path.join(tempfile.gettempdir(), "wind_farm_state.json"),
)


def write_wind_state(speed: float, direction_deg: float, path: str = DEFAULT_PATH) -> None:
    """Write the current wind (speed [m/s], direction [deg]) for other processes to read."""
    payload = {"speed": float(speed), "direction_deg": float(direction_deg) % 360.0}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)   # atomic on POSIX and Windows


def read_wind_state(path: str = DEFAULT_PATH,
                     default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Read the current (speed, direction_deg); returns `default` if nothing's been written yet."""
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return float(payload["speed"]), float(payload["direction_deg"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return default
