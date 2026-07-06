import time

import numpy as np
import math

from main1 import main

# SemanticAnnotation is only available with the framework installed.
# Guarded so queries.py also imports standalone (e.g. just to compare farms).
try:
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
except Exception:  # noqa: BLE001
    SemanticAnnotation = "SemanticAnnotation"  # type: ignore

# Pure formulas + the two farm definitions (no framework dependency).
from turbine_formulas import min_wind_speed_for_length, real_efficiency, wind_power_for_length
from wind_farm_export import R_BLADE_LENGTH, TurbineSpec



def _blade_length(spec: TurbineSpec) -> float:
    """Blade length L for a turbine spec."""
    return spec.rotor_blade_length or (spec.tower_height * R_BLADE_LENGTH)


def hub_wind_speed(ground_wind_speed: float, hub_height: float,
                   ref_height: float = 10.0, alpha: float = 0.0) -> float:
    """Wind speed at hub height via the power-law profile  v = v0*(h/h0)^alpha.

    alpha = 0 (default) -> no wind shear, every turbine sees ground_wind_speed.
    Taller turbines see more wind for alpha > 0 (typical onshore alpha ~ 0.14).
    """
    if hub_height <= 0 or ref_height <= 0:
        return ground_wind_speed
    return ground_wind_speed * (hub_height / ref_height) ** alpha


def farm_power(farm: list[TurbineSpec], rho: float = 1.225, wind_speed: float = 8.0,
               c_p: float = 0.45, alpha: float = 0.0, ref_height: float = 10.0) -> float:
    """Total generated power [W] of a farm = sum over turbines of generated power.

    A turbine below its cut-in wind speed is not spinning and contributes 0.
    """
    total = 0.0
    eff = real_efficiency(c_p)
    for spec in farm:
        L = _blade_length(spec)
        v = hub_wind_speed(wind_speed, spec.tower_height, ref_height, alpha)
        if abs(v) < min_wind_speed_for_length(L):
            continue                       # below cut-in -> no spin -> no power
        total += eff * wind_power_for_length(rho, v, L)
    return total



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


# world, driver = main()
# time.sleep(5)   # let the 20 Hz timer step the driver and ramp RPM up
# print(turbine_status(driver, "Farm_East_1"))
# print(turbine_status(driver, "Farm_North_1"))
# print(turbine_status(driver, "Farm_West_1"))
# print(turbine_status(driver, "Farm_South_1"))