# SemanticAnnotation is only available with the framework installed.
# Guarded so queries.py also imports standalone (e.g. just to compare farms).
import time

from history_file import clear_history, load_history, DEFAULT_HISTORY_FILE
from main1 import main
from peak_state_file import clear_peak_state

try:
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
except Exception:  # noqa: BLE001
    SemanticAnnotation = "SemanticAnnotation"  # type: ignore


# ------------------------------------------------------------------ #
# LIVE queries against a running SemanticWindDriver (main1.py)
# ------------------------------------------------------------------ #
def is_turbine_spinning(driver, name: str, eps: float = 0.05) -> bool:
    """True if turbine `name`'s rotor is currently turning (RPM above a small threshold)."""
    return driver.is_spinning(name, eps=eps)


def turbine_rpm(driver, name: str) -> float:
    """Current (ramped) RPM of turbine `name`, live from the running driver."""
    return driver.rpm(name)


def turbine_nacelle_yaw_deg(driver, name: str) -> float:
    """Current nacelle yaw angle of turbine `name`, in degrees, live from the running driver."""
    return driver.nacelle_yaw_deg(name)


def turbine_status(driver, name: str = None) -> dict:
    """Full live snapshot: wind speed/direction plus one turbine (or all of them)."""
    return driver.status(name)


def fastest_turbine(driver) -> (str, float):
    """Name and RPM of the turbine with the fastest rotor speed."""
    return (max(driver.turbines.values(), key=lambda t: t.current_rpm).name, max(driver.turbines.values(), key=lambda t: t.current_rpm).current_rpm)


def slowest_turbine(driver) -> (str, float):
    """Name and RPM of the turbine with the slowest rotor speed."""
    return (min(driver.turbines.values(), key=lambda t: t.current_rpm).name, min(driver.turbines.values(), key=lambda t: t.current_rpm).current_rpm)

def slowest_moving_turbine(driver) -> (str, float):
    """Name and RPM of the moving turbine with the slowest rotor speed."""
    moving = [t for t in driver.turbines.values() if abs(t.current_rpm) > 1e-9]
    if not moving:
        return (None, 0.0)
    turbine = min(moving, key=lambda t: abs(t.current_rpm))
    return (turbine.name, turbine.current_rpm)

def most_powerful_turbine(driver) -> (str, float):
    """Name and generated power [W] of the turbine producing the most."""
    generating = [t for t in driver.turbines.values() if t.current_power > 0]
    if not generating:
        return (None, 0.0)
    turbine = max(generating, key=lambda t: t.current_power)
    return (turbine.name, turbine.current_power)

def least_powerful_turbine(driver) -> (str, float):
    """Name and generated power [W] of the turbine producing the least."""
    return (min(driver.turbines.values(), key=lambda t: t.current_power).name, min(driver.turbines.values(), key=lambda t: t.current_power).current_power)

def least_powerful_moving_turbine(driver) -> (str, float):
    """Name and generated power [W] of the moving turbine producing the least."""
    moving = [t for t in driver.turbines.values() if abs(t.current_rpm) > 1e-9]
    if not moving:
        return (None, 0.0)
    turbine = min(moving, key=lambda t: abs(t.current_power))
    return (turbine.name, turbine.current_power)

def peak_power(driver) -> (str, float):
    """Timestamp of the highest power generation, and how much it was [W]."""
    if driver.peak_status is None:
        return (None, 0.0)
    return (driver.peak_status["timestamp"], driver.peak_status["total_power_w"])

def peak_power_report(driver) -> str:
    """Human-readable one-liner for the peak."""
    if driver.peak_status is None:
        return "no power has been generated yet"
    s = driver.peak_status
    return (f"peak {s['total_power_w']/1e6:.3f} MW at {s['timestamp']} "
            f"(t={s['time_s']:.2f}s, wind {s['wind_speed']:.1f} m/s "
            f"from {s['wind_direction_deg']:.0f} deg)")

# ------------------------------------------------------------------ #
# History Queries
# ------------------------------------------------------------------ #
def _samples(source):
    """Normalise `source` to a list of history samples.

    source may be: None (default file), a path str, a HistoryLog, or any
    object exposing a `.history` list. Returns the list of sample dicts.
    """
    if source is None:
        source = DEFAULT_HISTORY_FILE
    if isinstance(source, str):
        return load_history(source).history
    if hasattr(source, "history"):
        return source.history
    if isinstance(source, list):
        return source
    raise TypeError(f"cannot read history from {type(source).__name__}; "
                    f"pass a file path, a HistoryLog, or leave blank")

# --- 1.when was turbine X spinning? ---------------------------------  #
def spinning_intervals(name: str, source=None, eps: float = 0.05,
                       max_gap_s: float = 2.0) -> list[dict]:
    """Every continuous stretch of time turbine `name` was spinning.

    Reads the history file (see module docstring for `source`). Returns a list of
    {'start_s', 'end_s', 'duration_s', 'start_time', 'end_time'}. Empty list means
    it never span during the recorded history.

    max_gap_s guards against stitching across periods when NOTHING was recorded
    (e.g. between two runs). If consecutive samples are further apart than this,
    the open interval is closed at the last sample before the gap.
    """
    history = _samples(source)
    intervals, current, prev = [], None, None

    def close(at):
        current.update(end_s=at["time_s"], end_time=at["timestamp"],
                       duration_s=at["time_s"] - current["start_s"])
        intervals.append(current)

    for s in history:
        if (current is not None and prev is not None
                and s["time_s"] - prev["time_s"] > max_gap_s):
            close(prev)
            current = None

        spinning = abs(s["rpm"].get(name, 0.0)) > eps
        if spinning and current is None:
            current = {"start_s": s["time_s"], "start_time": s["timestamp"]}
        elif not spinning and current is not None:
            close(s)
            current = None
        prev = s

    if current is not None:  # still spinning at the end
        close(history[-1])
    return intervals


def was_spinning_at(name: str, time_s: float, source=None, eps: float = 0.05) -> bool:
    """Was turbine `name` spinning at time `time_s`? (nearest recorded sample)"""
    history = _samples(source)
    if not history:
        return False
    s = min(history, key=lambda s: abs(s["time_s"] - time_s))
    return abs(s["rpm"].get(name, 0.0)) > eps


# --- 2. what was the highest wind speed? ----------------------------- #
def highest_wind_speed(source=None) -> (str, float):
    """Timestamp of the highest recorded wind speed, and the speed [m/s]."""
    history = _samples(source)
    if not history:
        return (None, 0.0)
    s = max(history, key=lambda s: s["wind_speed"])
    return (s["timestamp"], s["wind_speed"])


# --- 3. at what wind speed did I have X MW? -------------------------- #
def wind_speed_for_power(target_mw: float, source=None, tol_frac: float = 0.05) -> dict:
    """Wind speed(s) at which the farm produced about `target_mw` megawatts.

    tol_frac is a relative tolerance (0.05 = within 5%). If nothing falls
    inside it, returns the closest sample found instead, with exact=False.
    """
    history = _samples(source)
    if not history:
        return {"exact": False, "wind_speeds": [], "note": "no history recorded"}
    target_w = target_mw * 1e6
    hits = [s for s in history
            if abs(s["total_power_w"] - target_w) <= tol_frac * target_w]
    if hits:
        return {
            "exact": True,
            "wind_speeds": sorted({round(s["wind_speed"], 2) for s in hits}),
            "first_time": hits[0]["timestamp"],
            "first_time_s": hits[0]["time_s"],
            "n_samples": len(hits),
        }
    closest = min(history, key=lambda s: abs(s["total_power_w"] - target_w))
    return {
        "exact": False,
        "wind_speeds": [round(closest["wind_speed"], 2)],
        "closest_mw": closest["total_power_w"] / 1e6,
        "first_time": closest["timestamp"],
        "first_time_s": closest["time_s"],
        "note": f"nothing within {tol_frac:.0%}; showing closest sample",
    }


world, driver = main()
time.sleep(5)   # let the 20 Hz timer step the driver and ramp RPM up
# print(turbine_status(driver, "Farm_East_1"))
# print(is_turbine_spinning(driver, "Farm_East_1"))
# print(turbine_rpm(driver, "Farm_East_1"))
# print(turbine_nacelle_yaw_deg(driver, "Farm_East_1"))
# print(fastest_turbine(driver))
# print(slowest_turbine(driver))
# print(slowest_moving_turbine(driver))
# print("---------------------------------------------")
# print(most_powerful_turbine(driver))
# print(least_powerful_turbine(driver))
# print(least_powerful_moving_turbine(driver))
# print("---------------------------------------------")
# print(peak_power(driver))


"""
spinning_intervals("Farm_East_tall")   # [{'start_time','end_time','duration_s',...}]
was_spinning_at("Farm_East_tall", 9.0) # True/False at a sim time
highest_wind_speed()                      # ('2026-07-23T21:09:51', 18.0)
wind_speed_for_power(10.0)              # wind speed(s) that produced ~10 MW
"""

#print(wind_speed_for_power(driver, 10.0))
#print(wind_speed_for_power(driver, 10.0))
print(spinning_intervals("Farm_East_tall"))
print(was_spinning_at("Farm_East_tall", 15.0))
print(highest_wind_speed())
print(wind_speed_for_power(13.300000))

# print("samples:", len(driver.history))
# if driver.history:
#     s = driver.history[-1]
#     print("last sample:", s["time_s"], s["wind_speed"], "m/s")
#     print("keys in rpm:", list(s["rpm"])[:3], "...")
#     print("max tall rpm seen:", max(abs(x["rpm"].get("Farm_East_tall", 0.0))
#                                     for x in driver.history))

# clear_peak_state()
# clear_history()