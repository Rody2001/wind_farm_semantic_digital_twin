"""
turbine_formulas.py
===================================================================
Pure turbine formulas, shared by:
  - queries.py            (digital-twin / semantic side)
  - wind_turbine_sim.py   (MuJoCo side)

No framework imports (numpy only).

    RPM   = 60 * v * TSR / (pi * 2 * L)
    v_min = (pi * 2 * L) / (60 * TSR)          # wind speed at exactly 1 RPM

    P_avail = 0.5 * rho * v^3 * (pi * L^2)     # power in the wind through the disc
    P_gen   = real_efficiency(c_p) * P_avail   # what the turbine actually makes
===================================================================
"""

import numpy as np

TSR = 6.0          # tip-speed ratio (blade-tip speed / wind speed)
RHO = 1.225        # air density [kg/m^3]
C_P = 0.45         # power coefficient (before drivetrain/other losses)


# ---- rotation ----------------------------------------------------- #
def min_wind_speed_for_length(blade_length: float, tsr: float = TSR) -> float:
    """Cut-in wind speed [m/s]: the speed at which the rotor turns at 1 RPM."""
    return (np.pi * 2 * blade_length) / (60 * tsr)


def rpm_for_wind(wind_speed: float, blade_length: float, tsr: float = TSR) -> float:
    """RPM = 60 * v * TSR / (pi * 2 * L); 0 below the 1-RPM cut-in speed."""
    if abs(wind_speed) < min_wind_speed_for_length(blade_length, tsr):
        return 0.0
    return (60 * wind_speed * tsr) / (np.pi * 2 * blade_length)


# ---- power -------------------------------------------------------- #
def real_efficiency(c_p: float = C_P, k_m: float = 0.015, k_e: float = 0.0125,
                    k_et: float = 0.065, k_t: float = 0.025, k_w: float = 0.0) -> float:
    """Overall efficiency = c_p times the drivetrain/other loss factors."""
    return (1 - k_m) * (1 - k_e) * (1 - k_et) * (1 - k_t) * (1 - k_w) * c_p


def wind_power_for_length(rho: float, wind_speed: float, blade_length: float) -> float:
    """Available wind power [W] through the rotor disc: 0.5 * rho * v^3 * (pi * L^2)."""
    return 0.5 * rho * abs(wind_speed) ** 3 * (np.pi * blade_length ** 2)


def generated_power_for_length(rho: float, wind_speed: float, blade_length: float,
                               c_p: float = C_P) -> float:
    """Generated electrical power [W] for one turbine."""
    return real_efficiency(c_p) * wind_power_for_length(rho, wind_speed, blade_length)