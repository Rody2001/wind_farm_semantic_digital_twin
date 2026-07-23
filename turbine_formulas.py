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
RHO = 1.225        # air density [kg/m^3] -- standard conditions, see rho_for_temperature(15.0)
C_P = 0.45         # power coefficient (before drivetrain/other losses)


# ---- air density ----------------------------------------------------- #
def rho_for_temperature(temp_c: float) -> float:
    """Air density [kg/m^3] from temperature via the ideal gas law.

    rho = M / (R_specific * T), with M = molar mass of dry air [kg/mol],
    R_specific in the same units ENERCON's datasheet uses, T in Kelvin.
    rho_for_temperature(15.0) ~= 1.225 kg/m^3, matching the RHO constant
    above and the "Standardluftdichte" in the ENERCON E-138 datasheet
    (1.225 kg/m^3 at 15 degC).
    """
    return (28.97 * 10**(-3)) / (8.2056 * 10**(-5) * (273.15 + temp_c))


# ---- wind vector ---------------------------------------------------- #
def wind_from_bearing(speed: float, bearing_deg: float) -> np.ndarray:
    """World-frame wind vector for wind coming FROM `bearing_deg` (0-360, compass bearing).

    Map: +X = East, +Y = North. bearing is measured clockwise from North, matching
    a real compass (0=N, 90=E, 180=S, 270=W). "from N" (0 deg) blows toward -Y (south).
    Shared by the MuJoCo sim and the semantic-digital-twin driver so both worlds agree.
    """
    a = np.deg2rad(bearing_deg)
    source = np.array([np.sin(a), np.cos(a), 0.0])   # unit vector toward where wind comes from
    return -source * speed


# ---- rotation ----------------------------------------------------- #
def min_wind_speed_for_length(blade_length: float, tsr: float = TSR) -> float:
    """Cut-in wind speed [m/s]: the speed at which the rotor turns at 1 RPM."""
    return (np.pi * 2 * blade_length) / (60 * tsr)


def rpm_for_wind(wind_speed: float, blade_length: float, tsr: float = TSR) -> float:
    """RPM = 60 * v * TSR / (pi * 2 * L); 0 below the 1-RPM cut-in speed."""
    if abs(wind_speed) < min_wind_speed_for_length(blade_length, tsr) or abs(wind_speed) >= 28.0:
        return 0.0
    return (60 * wind_speed * tsr) / (np.pi * 2 * blade_length)


# ---- power coefficient ---------------------------------------------- #
BETZ_LIMIT = 16.0 / 27.0   # ~0.5926, the physical maximum any turbine's cp can reach


def cp_for_wind(wind_speed: float) -> float:
    """Power coefficient cp(v) from a 6th-order polynomial fit to measured data.

        cp(v) = 2.181e-6 v^6 - 9.334e-5 v^5 + 1.278e-3 v^4 - 2.108e-3 v^3
                - 9.360e-2 v^2 + 0.7076 v - 1.0154

    This matches a realistic cp curve (rises to a peak around v=5 m/s, then
    tapers off) roughly over v in [2, 15] m/s, the range it was fit to. Outside
    that -- especially above ~18 m/s -- the raw polynomial diverges wildly
    (it reaches cp > 100 by 28 m/s), which is physically impossible: cp can
    never exceed the Betz limit (~0.593) or go negative. Both are clamped here.
    """
    v_1 = abs(wind_speed)
    cp = (2.181e-6 * v_1**6 - 9.334e-5 * v_1**5 + 1.278e-3 * v_1**4 - 2.108e-3 * v_1**3
          - 9.360e-2 * v_1**2 + 0.7076 * v_1 - 1.0154)
    return float(np.clip(cp, 0.0, BETZ_LIMIT))
    #return 0.45  # for testing, always return a fixed value

# ---- power -------------------------------------------------------- #
def real_efficiency(c_p: float = C_P, k_m: float = 0.015, k_e: float = 0.0125,
                    k_et: float = 0.065, k_t: float = 0.025, k_w: float = 0.0) -> float:
    """Overall efficiency = c_p times the drivetrain/other loss factors."""
    return (1 - k_m) * (1 - k_e) * (1 - k_et) * (1 - k_t) * (1 - k_w) * c_p


def wind_power_for_length(rho: float, wind_speed: float, blade_length: float) -> float:
    """Available wind power [W] through the rotor disc: 0.5 * rho * v^3 * (pi * L^2)."""
    return 0.5 * rho * abs(wind_speed) ** 3 * (np.pi * blade_length ** 2)


def generated_power_for_length(rho: float, wind_speed: float, blade_length: float,
                               c_p: float = None) -> float:
    """Generated electrical power [W] for one turbine.

    By default cp is taken from cp_for_wind(wind_speed) -- the polynomial fit --
    so power correctly follows the measured cp curve instead of using one fixed
    coefficient for every wind speed. Pass c_p explicitly to override with a
    fixed coefficient instead (e.g. for quick back-of-envelope estimates).
    """
    cp = cp_for_wind(wind_speed) if c_p is None else c_p
    return real_efficiency(cp) * wind_power_for_length(rho, wind_speed, blade_length)