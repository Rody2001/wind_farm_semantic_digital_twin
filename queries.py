# SemanticAnnotation is only available with the framework installed.
# Guarded so queries.py also imports standalone (e.g. just to compare farms).
import time

from main1 import main

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


world, driver = main()
time.sleep(5)   # let the 20 Hz timer step the driver and ramp RPM up
print(turbine_status(driver, "Farm_East_1"))
print(turbine_status(driver, "Farm_North_1"))
print(turbine_status(driver, "Farm_West_1"))
print(turbine_status(driver, "Farm_South_1"))