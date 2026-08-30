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

Per-turbine rating (see capped_power_for_length):

    v < v_rated      -> P_gen as above
    v >= v_rated     -> max_kw (the turbine's rated output, held flat)
    v >= v_cut_out   -> 0       (the turbine shuts down)

Cut-out, rating and rated speed all scale with blade length, the same way the
geometry in wind_farm_export.py scales with tower height. The reference turbine
is the 69 m blade, which cuts out at 28 m/s and is rated 2240 kW from 12.5 m/s:

    v_cut_out = L * (28   / 69)
    max_kw   = L * (2240 / 69)
    v_rated   = L * (12.5 / 69)
===================================================================
"""

import numpy as np

TSR = 4.19          # tip-speed ratio (blade-tip speed / wind speed)
RHO = 1.225        # air density [kg/m^3] -- standard conditions, see rho_for_temperature(15.0)
C_P = 0.45         # power coefficient (before drivetrain/other losses)
KW_TO_W = 1000.0    # max_kw is given in kW; every power value here is in W

# ---- the reference turbine every ratio below is taken from ----------- #
REF_BLADE_LENGTH = 69.0        # [m]
CUT_OUT_SPEED_REF = 28.0       # [m/s]  cut-out of the reference turbine
max_kw_REF = 2240.0           # [kW]   rated output of the reference turbine
max_kw_WIND_SPEED_REF = 12.5  # [m/s]  rated wind speed of the reference turbine

R_CUT_OUT = CUT_OUT_SPEED_REF / REF_BLADE_LENGTH             # 28   / 69
R_max_kw = max_kw_REF / REF_BLADE_LENGTH                   # 2240 / 69
R_max_kw_WIND_SPEED = max_kw_WIND_SPEED_REF / REF_BLADE_LENGTH   # 12.5 / 69


# ---- per-turbine ratings, all derived from blade length --------------- #
def cut_out_for_length(blade_length: float) -> float:
    """Cut-out wind speed [m/s]: at and above this the rotor is stopped (RPM = 0, power = 0)."""
    return abs(blade_length) * R_CUT_OUT


def max_kw_for_length(blade_length: float) -> float:
    """Rated output [kW] this turbine holds once the wind reaches its rated speed."""
    return abs(blade_length) * R_max_kw


def max_kw_wind_speed_for_length(blade_length: float) -> float:
    """Rated wind speed [m/s]: where the power curve flattens off at max_kw."""
    return abs(blade_length) * R_max_kw_WIND_SPEED


def rating_for_length(blade_length: float, max_kw: float = 0.0,
                      max_kw_wind_speed: float = 0.0) -> tuple:
    """Resolve (max_kw, max_kw_wind_speed) for one turbine.

    0 (or None) means "derive it from the blade length", the same convention
    rotor_blade_length=0 already uses for "derive it from the tower height".
    Pass a non-zero value to override that turbine's rating by hand.
    """
    return (float(max_kw) if max_kw else max_kw_for_length(blade_length),
            float(max_kw_wind_speed) if max_kw_wind_speed
            else max_kw_wind_speed_for_length(blade_length))


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
    """RPM = 60 * v * TSR / (pi * 2 * L).

    0 below the 1-RPM cut-in speed, and 0 again at and above this turbine's own
    cut-out speed, which scales with blade length: cut_out_for_length(L).
    """
    if (abs(wind_speed) < min_wind_speed_for_length(blade_length, tsr)
            or abs(wind_speed) >= cut_out_for_length(blade_length)):
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
    # return 0.45  # for testing, always return a fixed value

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


def capped_power_for_length(rho: float, wind_speed: float, blade_length: float,
                            max_kw: float = 0.0, max_kw_wind_speed: float = 0.0,
                            c_p: float = None) -> float:
    """Generated electrical power [W] for one turbine, capped at its rated output.

    This is the generalisation of the old hard-coded conditioned_power_out_put():
    instead of "2240 kW above 12.5 m/s, cut out at 28 m/s" for every turbine, all
    three numbers scale with that turbine's blade length.

        v >= cut_out_for_length(L)  ->  0        (rotor stopped, checked first)
        v >= max_kw_wind_speed     ->  max_kw  (rated output, held flat)
        otherwise                   ->  generated_power_for_length(...)

    Args:
        max_kw: rated output [kW]. 0 (or None) derives it from the blade
            length: L * (2240 / 69).
        max_kw_wind_speed: rated wind speed [m/s]. 0 (or None) derives it from
            the blade length: L * (12.5 / 69).

    Returns W, like every other power function in this module, so max_kw is
    multiplied by KW_TO_W on the way out.
    """
    v = abs(wind_speed)
    if v >= cut_out_for_length(blade_length):
        return 0.0

    max_kw, rated_v = rating_for_length(blade_length, max_kw, max_kw_wind_speed)
    if v >= rated_v:
        return max_kw * KW_TO_W
    return generated_power_for_length(rho, wind_speed, blade_length, c_p)


def conditioned_power_out_put(rho: float, wind_speed: float, blade_length: float,
                               c_p: float = None) -> float:
    """Backwards-compatible shim: the old fixed 2240 kW / 12.5 m/s / 28 m/s numbers.

    Superseded by capped_power_for_length(), where all three scale with blade
    length. Kept so older call sites keep working; note it returns the raw 2240
    (not 2240 kW in W) exactly as before, and ignores blade length entirely.
    """
    v = abs(wind_speed)
    if v >= CUT_OUT_SPEED_REF:
        return 0.0
    if v >= max_kw_WIND_SPEED_REF:
        return max_kw_REF
    return generated_power_for_length(rho, wind_speed, blade_length, c_p)