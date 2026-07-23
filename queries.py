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


world, driver = main()
time.sleep(5)   # let the 20 Hz timer step the driver and ramp RPM up
# print(turbine_status(driver, "Farm_East_1"))
# print(is_turbine_spinning(driver, "Farm_East_1"))
# print(turbine_rpm(driver, "Farm_East_1"))
# print(turbine_nacelle_yaw_deg(driver, "Farm_East_1"))
print(fastest_turbine(driver))
print(slowest_turbine(driver))
print(slowest_moving_turbine(driver))
print("---------------------------------------------")
print(most_powerful_turbine(driver))
print(least_powerful_turbine(driver))
print(least_powerful_moving_turbine(driver))
print("---------------------------------------------")
#print(vars(next(iter(driver.turbines.values()))))
# print(driver.turbines.values())