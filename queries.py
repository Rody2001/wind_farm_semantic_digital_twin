"""
queries.py
===================================================================
Queries over the semantic digital twin.

Every live query below reads a SemanticAnnotation out of the World. Nothing here
touches driver.turbines or driver.wind_speed:

    is_turbine_spinning(driver, "Farm_East_tall")
        -> finds the RotorSpeed annotation whose turbine is "Farm_East_tall"
           (it is rooted at that turbine's hub Body)
        -> returns True if its value is above the threshold

The queries accept either the World or the driver -- the driver only serves as a
handle to reach its .world, so the existing call sites keep working unchanged.

The annotations themselves are written by SemanticWindDriver.step(), in the same
loop that moves the turbines, so they always show what MuJoCo is doing now.

The history queries at the bottom are different in kind: they answer questions
about time spans in a run that has already happened, which the world cannot hold
because the world only ever represents the present. They read the 1 Hz log.
===================================================================
"""

import time

from history_file import load_history, DEFAULT_HISTORY_FILE
from main1 import main

from semantic_annotations import (
    GeneratedEnergy, GeneratedPower, NacelleYaw, RotorSpeed,
    Temperature, WindDirection, WindSpeed, Hub, Tower,
)

try:
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
except Exception:  # noqa: BLE001
    SemanticAnnotation = "SemanticAnnotation"  # type: ignore


# ------------------------------------------------------------------ #
# ENVIRONMENT queries -- the three environment annotations
# ------------------------------------------------------------------ #
def wind_speed(source) -> float:
    """Current wind speed [m/s], from the world's WindSpeed annotation."""
    return source.get_semantic_annotations_by_type(WindSpeed)[0].value

def wind_direction(source) -> float:
    """Compass bearing the wind comes FROM [deg], from the WindDirection annotation."""
    return source.get_semantic_annotations_by_type(WindDirection)[0].value


def temperature(source) -> float:
    """Current air temperature [deg C], from the Temperature annotation."""
    return source.get_semantic_annotations_by_type(Temperature)[0].value


def environment(source) -> dict:
    """All three environment annotations in one dict."""
    return {
        "wind_speed": wind_speed(source),
        "wind_direction_deg": wind_direction(source),
        "temperature_c": temperature(source),
    }


def is_environment_live(source, max_age_s: float = 2.0) -> bool:
    """Is the world still being updated, or are these numbers stale?"""
    ann = source.get_semantic_annotations_by_type(WindSpeed)[0]
    return ann is not None and ann.is_fresh(max_age_s)


def annotation_report(source) -> str:
    """One line per measured annotation -- handy for a screenshot in the thesis."""
    lines = [f"  {source.get_semantic_annotations_by_type(cls)[0]}"
             for cls in (WindSpeed, WindDirection, Temperature)]
    all_rotor_speeds = source.get_semantic_annotations_by_type(RotorSpeed)
    for ann in all_rotor_speeds:
        lines.append(f"  {ann}")
    all_turbine_yaws = source.get_semantic_annotations_by_type(NacelleYaw)
    for ann in all_turbine_yaws:
        lines.append(f"  {ann}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# LIVE turbine queries -- the per-turbine annotations in the world
# ------------------------------------------------------------------ #
def turbine_names(source) -> list:
    """Every turbine that has a RotorSpeed annotation in the world."""
    all_rotor_speeds = source.get_semantic_annotations_by_type(RotorSpeed)
    return [a.turbine for a in all_rotor_speeds]


def turbine_rpm(source, name: str) -> float:
    """Rotor speed [rpm] of turbine `name`, from the RotorSpeed on its hub."""
    # return scalar(source, RotorSpeed, name)
    all_rotor_speeds = source.get_semantic_annotations_by_type(RotorSpeed)
    for rpm in all_rotor_speeds:
        if rpm.turbine == name:
            return rpm.value

def turbine_power(source, name: str) -> float:
    """Generated power [W] of turbine `name`, from its GeneratedPower annotation."""
    # return scalar(source, GeneratedPower, name)
    all_turbine_powers = source.get_semantic_annotations_by_type(GeneratedPower)
    for power in all_turbine_powers:
        if power.turbine == name:
            return power.value


def turbine_energy(source, name: str) -> float:
    """Energy generated so far [kWh] by turbine `name`."""
    # return scalar(source, GeneratedEnergy, name)
    all_turbine_energies = source.get_semantic_annotations_by_type(GeneratedEnergy)
    for energy in all_turbine_energies:
        if energy.turbine == name:
            return energy.value


def turbine_nacelle_yaw_deg(source, name: str) -> float:
    """Nacelle yaw angle [deg] of turbine `name`, from the NacelleYaw on its nacelle."""
    # return scalar(source, NacelleYaw, name)
    all_turbine_yaws = source.get_semantic_annotations_by_type(NacelleYaw)
    for yaw in all_turbine_yaws:
        if yaw.turbine == name:
            return yaw.value


def is_turbine_spinning(source, name: str, eps: float = 0.05) -> bool:
    """True if turbine `name`'s rotor is turning (RPM above a small threshold)."""
    return abs(turbine_rpm(source, name)) > eps


def turbine_body(source, name: str):
    """The Body the turbine's rotor speed is attached to -- its hub."""
    ann1 = [
        a for a in source.get_semantic_annotations_by_type(RotorSpeed)
        if a.turbine == name
    ]
    return None if ann1[0] is None else ann1[0].root


def turbine_status(source, name: str = None) -> dict:
    """Full snapshot: the environment plus one turbine, or every turbine."""
    if name is not None:
        return {
            "name": name,
            "rpm": turbine_rpm(source, name),
            "spinning": is_turbine_spinning(source, name),
            "nacelle_yaw_deg": turbine_nacelle_yaw_deg(source, name),
            "power_w": turbine_power(source, name),
            "energy_kwh": turbine_energy(source, name),
            **environment(source),
        }
    return {
        **environment(source),
        "turbines": {n: turbine_status(source, n) for n in turbine_names(source)},
    }


def total_power(source) -> float:
    """Total generated power [W] across the whole farm right now."""
    return sum(a.value for a in source.get_semantic_annotations_by_type(GeneratedPower))


def total_energy(source) -> float:
    """Total energy [kWh] generated across the farm since the run started."""
    return sum(a.value for a in source.get_semantic_annotations_by_type(GeneratedEnergy))


def spinning_turbines(source, eps: float = 0.05) -> list:
    """Names of every turbine currently turning."""
    return [a.turbine for a in source.get_semantic_annotations_by_type(RotorSpeed) if abs(a.value) > eps]


def idle_turbines(source, eps: float = 0.05) -> list:
    """Names of every turbine standing still: below cut-in, or not facing the wind."""
    return [a.turbine for a in source.get_semantic_annotations_by_type(RotorSpeed) if abs(a.value) <= eps]


def fastest_turbine(source) -> (str, float):
    """Name and RPM of the turbine with the fastest rotor speed."""
    anns = source.get_semantic_annotations_by_type(RotorSpeed)
    if not anns:
        return (None, 0.0)
    a = max(anns, key=lambda a: a.value)
    return (a.turbine, a.value)


def slowest_turbine(source) -> (str, float):
    """Name and RPM of the turbine with the slowest rotor speed."""
    anns = source.get_semantic_annotations_by_type(RotorSpeed)
    if not anns:
        return (None, 0.0)
    a = min(anns, key=lambda a: a.value)
    return (a.turbine, a.value)


def slowest_moving_turbine(source) -> (str, float):
    """Name and RPM of the moving turbine with the slowest rotor speed."""
    moving = [a for a in source.get_semantic_annotations_by_type(RotorSpeed) if abs(a.value) > 1e-9]
    if not moving:
        return (None, 0.0)
    a = min(moving, key=lambda a: abs(a.value))
    return (a.turbine, a.value)


def most_powerful_turbine(source) -> (str, float):
    """Name and generated power [W] of the turbine producing the most."""
    generating = [a for a in source.get_semantic_annotations_by_type(GeneratedPower) if a.value > 0]
    if not generating:
        return (None, 0.0)
    a = max(generating, key=lambda a: a.value)
    return (a.turbine, a.value)


def least_powerful_turbine(source) -> (str, float):
    """Name and generated power [W] of the turbine producing the least."""
    anns = source.get_semantic_annotations_by_type(GeneratedPower)
    if not anns:
        return (None, 0.0)
    a = min(anns, key=lambda a: a.value)
    return (a.turbine, a.value)


def least_powerful_moving_turbine(source) -> (str, float):
    """Name and generated power [W] of the moving turbine producing the least."""
    moving = {a.turbine for a in source.get_semantic_annotations_by_type(RotorSpeed)
              if abs(a.value) > 1e-9}
    generating = [a for a in source.get_semantic_annotations_by_type(GeneratedPower)
                  if a.turbine in moving]
    if not generating:
        return (None, 0.0)
    a = min(generating, key=lambda a: a.value)
    return (a.turbine, a.value)


# ------------------------------------------------------------------ #
# PEAK queries -- a record of the past, so still driver state
# ------------------------------------------------------------------ #

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
def highest_wind_speed(source=None) -> (str, str, float):
    """Timestamp of the highest recorded wind speed, and the speed [m/s]."""
    history = _samples(source)
    if not history:
        return (None, 0.0)
    s = max(history, key=lambda s: s["wind_speed"])
    return (s["timestamp"],s["time_s"], s["wind_speed"])


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

def time_for_energy(target_kwh: float, source=None) -> dict:
    """The first moment the farm had produced at least `target_kwh` kilowatt-hours.

    Energy accumulates, so this is not a search for a matching value but for the
    first sample that reaches the target. Returns the time and the wind speed at
    that moment, or reached=False if the run ended before getting there.
    """
    history = _samples(source)
    if not history:
        return {"reached": False, "note": "no history recorded"}

    for s in history:
        if s.get("total_energy_kwh", 0.0) >= target_kwh:
            return {
                "reached": True,
                "target_kwh": target_kwh,
                "time_s": s["time_s"],
                "timestamp": s["timestamp"],
                "wind_speed": s["wind_speed"],
                "total_energy_kwh": s["total_energy_kwh"],
            }

    best = max(history, key=lambda s: s.get("total_energy_kwh", 0.0))
    return {
        "reached": False,
        "target_kwh": target_kwh,
        "highest_kwh": best.get("total_energy_kwh", 0.0),
        "time_s": best["time_s"],
        "note": "the run ended before reaching the target",
    }