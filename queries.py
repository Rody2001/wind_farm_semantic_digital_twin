import numpy as np

from main import main
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
import math




def wind_power(rho:float, wind_speed:float, blade: SemanticAnnotation) -> float:
    world = blade._world
    effect_area = np.pi * (blade.bodies[0].visual.scale.z)**2

    return (0.5 * rho * wind_speed**3 * effect_area)


def real_efficiency_calculater(c_p:float, k_m:float = 0.015, k_e:float = 0.0125, k_et:float = 0.065, k_t:float = 0.025, k_w:float = 0) -> float:
    real_efficiency = (1 - k_m) * (1 - k_e) * (1 - k_et) * (1 - k_t) * (1 - k_w) * c_p
    return real_efficiency


def generated_energy(rho:float, wind_speed:float, blade: SemanticAnnotation, c_p:float) -> float:
    return (real_efficiency_calculater(c_p) * wind_power(rho, wind_speed, blade))


def minimum_wind_speed(rho:float, blade: SemanticAnnotation, p_w:float = 0.1) -> float:
    effect_area = np.pi * (blade.bodies[0].visual.scale.z)**2
    min_speed = (p_w / (0.5 * rho * effect_area )) ** (1/3)
    return min_speed


world1 = main()
print(minimum_wind_speed(1.225, world1.get_semantic_annotation_by_name("rotor_blade1")))
#print(world1.get_semantic_annotation_by_name("rotor_blade1"))

# altitude = 100
# rho = 1.247015 * (math.exp(-0.000104 * altitude))
#
# Temperature = 20
# rho2 = (1013.25 / (287.1 * Temperature))