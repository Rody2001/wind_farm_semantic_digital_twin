"""
turbine_formulas.py
===================================================================
Pure turbine formulas, shared by:
  - queries.py            (digital-twin / semantic side)
  - wind_turbine_sim.py   (MuJoCo side)

No framework imports (numpy only), so the simulation can import this
without pulling in main.py / ROS / semantic_digital_twin.

    RPM   = 60 * v * TSR / (pi * 2 * L)        v = wind speed, L = blade length
    v_min = (pi * 2 * L) / (60 * TSR)          # the wind speed at exactly 1 RPM

TSR (tip-speed ratio) was the hard-coded "6" in the earlier version and
the explicit factor 2 makes L the rotor radius (tip speed = omega * L).
===================================================================
"""

import numpy as np

TSR = 6.0   # tip-speed ratio (ratio of blade-tip speed to wind speed)


def min_wind_speed_for_length(blade_length: float, tsr: float = TSR) -> float:
    """Cut-in wind speed [m/s]: the speed at which the rotor turns at 1 RPM.

    Inverse of rpm_for_wind at RPM = 1:  v = (pi * 2 * L) / (60 * TSR).
    """
    return (np.pi * 2 * blade_length) / (60 * tsr)


def rpm_for_wind(wind_speed: float, blade_length: float, tsr: float = TSR) -> float:
    """RPM = 60 * v * TSR / (pi * 2 * L).

    Returns 0.0 below the cut-in speed, so the turbine only spins when
    |wind_speed| >= min_wind_speed_for_length(blade_length).
    """
    if abs(wind_speed) < min_wind_speed_for_length(blade_length, tsr):
        return 0.0
    return (60 * wind_speed * tsr) / (np.pi * 2 * blade_length)