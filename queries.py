import numpy as np
import math

# SemanticAnnotation is only available with the framework installed.
# Guarded so queries.py also imports standalone (e.g. just to compare farms).
try:
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
except Exception:  # noqa: BLE001
    SemanticAnnotation = "SemanticAnnotation"  # type: ignore

# Pure formulas + the two farm definitions (no framework dependency).
from turbine_formulas import min_wind_speed_for_length
from wind_farm_export import WIND_FARM_A_small, WIND_FARM_B_big, R_BLADE_LENGTH, TurbineSpec


# ------------------------------------------------------------------ #
# single-turbine physics
# ------------------------------------------------------------------ #
def wind_power_from_length(rho: float, wind_speed: float, blade_length: float) -> float:
    """Available wind power [W] through the rotor disc: 0.5 * rho * v^3 * (pi * L^2)."""
    effect_area = np.pi * (blade_length ** 2)
    return 0.5 * rho * wind_speed ** 3 * effect_area


def wind_power(rho: float, wind_speed: float, blade: SemanticAnnotation) -> float:
    return wind_power_from_length(rho, wind_speed, blade.bodies[0].visual.scale.z)


def real_efficiency_calculater(c_p: float, k_m: float = 0.015, k_e: float = 0.0125,
                               k_et: float = 0.065, k_t: float = 0.025, k_w: float = 0) -> float:
    real_efficiency = (1 - k_m) * (1 - k_e) * (1 - k_et) * (1 - k_t) * (1 - k_w) * c_p
    return real_efficiency


def generated_energy(rho: float, wind_speed: float, blade: SemanticAnnotation, c_p: float) -> float:
    return (real_efficiency_calculater(c_p) * wind_power(rho, wind_speed, blade))


def minimum_wind_speed(blade: SemanticAnnotation) -> float:
    return min_wind_speed_for_length(blade.bodies[0].visual.scale.z)


# ------------------------------------------------------------------ #
# NEW: farm-level energy + comparison
# ------------------------------------------------------------------ #
def _blade_length(spec: TurbineSpec) -> float:
    """Blade length L for a turbine spec (same formula as main.py / the exporter)."""
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
    eff = real_efficiency_calculater(c_p)
    for spec in farm:
        L = _blade_length(spec)
        v = hub_wind_speed(wind_speed, spec.tower_height, ref_height, alpha)
        if abs(v) < min_wind_speed_for_length(L):
            continue                       # below cut-in -> no spin -> no power
        total += eff * wind_power_from_length(rho, v, L)
    return total


def which_farm_produces_more(rho: float = 1.225, wind_speed: float = 8.0,
                             c_p: float = 0.45, alpha: float = 0.0) -> dict:
    """Compare WIND_FARM_A_small vs WIND_FARM_B_big and report which makes more power."""
    pa = farm_power(WIND_FARM_A_small, rho, wind_speed, c_p, alpha)
    pb = farm_power(WIND_FARM_B_big, rho, wind_speed, c_p, alpha)
    winner = "A" if pa > pb else "B" if pb > pa else "tie"
    lo = min(pa, pb)
    return {
        "farm_A_W": pa,
        "farm_B_W": pb,
        "winner": winner,
        "ratio": (max(pa, pb) / lo) if lo > 0 else float("inf"),
    }


if __name__ == "__main__":
    # ---- farm comparison (no ROS needed) ----
    res = which_farm_produces_more(wind_speed=8.0, c_p=0.45)
    print(f"Farm A: {res['farm_A_W'] / 1e6:8.3f} MW   "
          f"({len(WIND_FARM_A_small)} turbines)")
    print(f"Farm B: {res['farm_B_W'] / 1e6:8.3f} MW   "
          f"({len(WIND_FARM_B_big)} turbines)")
    print(f"-> Farm {res['winner']} produces more "
          f"({res['ratio']:.2f}x) at 8 m/s")

#---- original digital-twin queries (need the framework) ----
# from main import main
# world1 = main()
# print(minimum_wind_speed(world1.get_semantic_annotation_by_name("wind2_rotor_blade1")))
# print(minimum_wind_speed(world1.get_semantic_annotation_by_name("wind3_rotor_blade1")))