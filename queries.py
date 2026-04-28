import numpy as np

from main import main
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
import math

from semantic_digital_twin.world import World


def q1(annotation: SemanticAnnotation, speed: float) -> bool:
    roh = 1204.1
    C_p = 0.593
    power = ...

    v_min = 0

    if v_min > speed:
        return True
    return False

#print(1.247015 * (e))
altitude = 100
roh = 1.247015 * (math.exp(-0.000104 * altitude))

# print(roh)

Temperature = 20
roh2 = (1013.25 / (287.1 * Temperature))

# print(roh2)

def generated_energy(efficiency : float, wind_speed: float, turbine_blade: SemanticAnnotation) -> float:

    world = turbine_blade._world
    affected_area = np.pi * (turbine_blade.bodies[0].visual.scale.z)**2
    density = 1.247015 * (math.exp(-0.000104 * world.get_body_by_name("tower").visual.scale.z)) # world.get_body_by_name("tower")

    return (0.5 * density * affected_area * wind_speed**3 * efficiency)


world = main()
print(generated_energy(1.22475007791, 0.59, 10, world.get_semantic_annotation_by_name("rotor_blade1")))

###