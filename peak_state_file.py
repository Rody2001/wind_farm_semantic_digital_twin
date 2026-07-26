"""
peak_state_file.py
===================================================================
Persist the all-time power peak to a small JSON file so it survives
process restarts. Without this, every new query script starts with
peak_power_w = 0 and can only report the maximum of its own run --
so a peak recorded at 8 m/s is lost as soon as you restart at 4 m/s.

Same idea as wind_state_file: a tiny file acts as the shared source
of truth between short-lived processes.
===================================================================
"""

import json
import os
import tempfile

DEFAULT_PEAK_FILE = "peak_state.json"


def read_peak_state(path: str = DEFAULT_PEAK_FILE):
    """Return the stored peak snapshot dict, or None if there isn't one yet."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_peak_state(snapshot: dict, path: str = DEFAULT_PEAK_FILE) -> None:
    """Atomically overwrite the stored peak snapshot."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh)
        os.replace(tmp, path)          # atomic: readers never see a half-written file
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def clear_peak_state(path: str = DEFAULT_PEAK_FILE) -> bool:
    """Forget the stored peak (start a fresh experiment).

    Returns True if a file was actually removed, False if none was found,
    and prints the ABSOLUTE path so a working-directory mismatch is obvious.
    """
    ap = os.path.abspath(path)
    if os.path.exists(ap):
        os.remove(ap)
        print(f"[clear_peak_state] deleted {ap}")
        return True
    print(f"[clear_peak_state] nothing to delete at {ap}")
    return False