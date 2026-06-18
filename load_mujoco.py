"""
Automatic MuJoCo loader for Wind Turbine Semantic Digital Twin.
Converts the semantic digital twin model to MuJoCo XML and runs it.
"""

import mujoco
import mujoco.viewer
import numpy as np
from dataclasses import dataclass, field

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
import semantic_digital_twin.spatial_types.spatial_types as cas

from semantic_annotations import Tower, Nacelle, RotorBlades, Hub, TowerBase


@dataclass
class WindTurbine:
    """Wind turbine builder for MuJoCo."""

    @classmethod
    def create_with_new_body_in_world(
        cls,
        world: World,
        parent: Body,
        name: PrefixedName,
        tower_height: float,
        rotor_blade_length: float = 0.0,
        parent_T_connection: HomogeneousTransformationMatrix = None,
    ):
        """Create a complete wind turbine structure with 3 rotor blades."""
        if parent_T_connection is None:
            parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0.2)

        if rotor_blade_length == 0:
            rotor_blade_length = tower_height*(8475/15797)

        elements_conns = []
        elements_annotations = []

        nacelle_length = tower_height * (1500/15797)

        # Define colors
        red = Color(1, 0, 0)
        white = Color(1, 1, 1)
        blue = Color(0, 0, 1)
        brown = Color(1, 0.5, 0.25)
        green = Color(0, 1, 0)

        # Tower Base
        base_box = Box(scale=Scale(tower_height/3, tower_height/3, 0.2), color=brown)
        base_body = Body(name=PrefixedName(f"{name}_tower_base"),
                        visual=ShapeCollection([base_box]),
                        collision=ShapeCollection([base_box]))
        base_conn = FixedConnection(parent=parent, child=base_body,
                                   parent_T_connection_expression=parent_T_connection)
        base_annotation = TowerBase(root=base_body)
        elements_conns.append(base_conn)
        elements_annotations.append(base_annotation)

        # Tower
        tower_width = tower_height * (901/15797)
        tower_cylinder = Cylinder(width=tower_width, height=tower_height, color=red)
        tower_body = Body(name=PrefixedName(f"{name}_tower"),
                         visual=ShapeCollection([tower_cylinder]),
                         collision=ShapeCollection([tower_cylinder]))
        tower_conn = FixedConnection(parent=base_body, child=tower_body,
                                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                                        x=0, y=0, z=tower_height/2 + 0.1))
        tower_annotation = Tower(root=tower_body)
        elements_conns.append(tower_conn)
        elements_annotations.append(tower_annotation)

        # Nacelle
        nacelle_height = tower_height * (678/15797)
        nacelle_box = Box(scale=Scale(nacelle_length, tower_width, nacelle_height), color=green)
        nacelle_body = Body(name=PrefixedName(f"{name}_nacelle"),
                           visual=ShapeCollection([nacelle_box]),
                           collision=ShapeCollection([nacelle_box]))
        nacelle_conn = FixedConnection(parent=tower_body, child=nacelle_body,
                                      parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                                          x=-nacelle_length/4, y=0, z=(tower_height/2) + (nacelle_height/2)))
        nacelle_annotation = Nacelle(root=nacelle_body)
        elements_conns.append(nacelle_conn)
        elements_annotations.append(nacelle_annotation)

        # Hub
        hub_box = Box(scale=Scale(tower_width, tower_width, nacelle_height), color=blue)
        hub_body = Body(name=PrefixedName(f"{name}_hub"),
                       visual=ShapeCollection([hub_box]),
                       collision=ShapeCollection([hub_box]))
        hub_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=nacelle_body, child=hub_body, axis=cas.Vector3.X(),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=nacelle_length/2 + (tower_width /2)))
        hub_annotation = Hub(root=hub_body)
        elements_conns.append(hub_conn)
        elements_annotations.append(hub_annotation)

        # Blade 1
        blade1_box = Box(scale=Scale(0.1, 0.2, rotor_blade_length), color=white)
        blade1_body = Body(name=PrefixedName(f"{name}_rotor_blade1"),
                          visual=ShapeCollection([blade1_box]),
                          collision=ShapeCollection([blade1_box]))
        blade1_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=hub_body, child=blade1_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=-4.5*tower_width, z=2.5*tower_width, roll=-np.pi*2/3),
            axis=cas.Vector3.Z())
        blade1_annotation = RotorBlades(root=blade1_body, name=PrefixedName(f"{name}_rotor_blade1"))
        elements_conns.append(blade1_conn)
        elements_annotations.append(blade1_annotation)

        # Blade 2
        blade2_box = Box(scale=Scale(0.1, 0.2, rotor_blade_length), color=white)
        blade2_body = Body(name=PrefixedName(f"{name}_rotor_blade2"),
                          visual=ShapeCollection([blade2_box]),
                          collision=ShapeCollection([blade2_box]))
        blade2_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=hub_body, child=blade2_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
               y=4.5*tower_width, z=2.5*tower_width, roll=np.pi*2/3),
            axis=cas.Vector3.Z())
        blade2_annotation = RotorBlades(root=blade2_body, name=PrefixedName(f"{name}_rotor_blade2"))
        elements_conns.append(blade2_conn)
        elements_annotations.append(blade2_annotation)

        # Blade 3
        blade3_box = Box(scale=Scale(0.1, 0.2, rotor_blade_length), color=white)
        blade3_body = Body(name=PrefixedName(f"{name}_rotor_blade3"),
                          visual=ShapeCollection([blade3_box]),
                          collision=ShapeCollection([blade3_box]))
        blade3_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=hub_body, child=blade3_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-(rotor_blade_length/2) - (nacelle_height/2)),
            axis=cas.Vector3.Z())
        blade3_annotation = RotorBlades(root=blade3_body, name=PrefixedName(f"{name}_rotor_blade3"))
        elements_conns.append(blade3_conn)
        elements_annotations.append(blade3_annotation)

        # Add all to world
        for conn in elements_conns:
            world.add_connection(conn)
        for annotation in elements_annotations:
            world.add_semantic_annotation(annotation)

        return hub_conn


def generate_mujoco_xml(world):
    """
    Generate MuJoCo XML from semantic digital twin world.
    """
    xml_parts = ['<mujoco model="wind_farm">']

    # Compiler settings
    xml_parts.append('  <compiler angle="radian" coordinate="local"/>')

    # Assets for materials
    xml_parts.append('  <asset>')
    xml_parts.append('    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>')
    xml_parts.append('    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>')
    xml_parts.append('  </asset>')

    # Worldbody
    xml_parts.append('  <worldbody>')
    xml_parts.append('    <light pos="0 0 30" dir="0 0 -1" directional="true"/>')
    xml_parts.append('    <geom name="floor" type="plane" size="50 50 .1" material="grid"/>')

    # Convert each body from the world to MuJoCo format
    for body in world.bodies:
        xml_parts.extend(_convert_body_to_xml(world, body, indent=4))

    xml_parts.append('  </worldbody>')
    xml_parts.append('</mujoco>')

    return '\n'.join(xml_parts)


def _convert_body_to_xml(world, body, indent=4):
    """Convert a semantic digital twin body to MuJoCo XML elements."""
    prefix = ' ' * indent
    xml_parts = []

    # Get connection to this body to determine position
    connection = None
    for conn in world.connections:
        if hasattr(conn, 'child') and conn.child == body:
            connection = conn
            break

    if connection and hasattr(connection, 'parent_T_connection_expression'):
        # Extract position from transformation matrix
        T = connection.parent_T_connection_expression
        pos = f"{float(T.x):.3f} {float(T.y):.3f} {float(T.z):.3f}"
    else:
        pos = "0 0 0"

    # Start body tag
    body_xml = f'{prefix}<body name="{body.name}"'
    if pos != "0 0 0":
        body_xml += f' pos="{pos}"'
    body_xml += '>'
    xml_parts.append(body_xml)

    # Add visual geometries
    if hasattr(body, 'visual') and body.visual:
        for i, shape in enumerate(body.visual.shapes):
            geom_xml = _convert_shape_to_geom(shape, f"{body.name}_geom_{i}", indent + 2)
            xml_parts.append(geom_xml)

    # Check for joints (if this body has a revolute connection)
    if connection:
        from semantic_digital_twin.world_description.connections import RevoluteConnection
        if isinstance(connection, RevoluteConnection):
            axis = "1 0 0"  # default
            if hasattr(connection, 'axis'):
                ax = connection.axis
                axis = f"{ax.x:.0f} {ax.y:.0f} {ax.z:.0f}"
            xml_parts.append(f'{prefix}  <joint name="{body.name}_hinge" type="hinge" axis="{axis}"/>')

    xml_parts.append(f'{prefix}</body>')

    return xml_parts


def _convert_shape_to_geom(shape, name, indent):
    """Convert a shape to MuJoCo geom element."""
    from semantic_digital_twin.world_description.geometry import Box, Cylinder

    prefix = ' ' * indent
    rgba = f"{shape.color.r:.2f} {shape.color.g:.2f} {shape.color.b:.2f} 1"

    if isinstance(shape, Box):
        size = f"{shape.scale.x/2:.3f} {shape.scale.y/2:.3f} {shape.scale.z/2:.3f}"
        return f'{prefix}<geom name="{name}" type="box" size="{size}" rgba="{rgba}"/>'
    elif isinstance(shape, Cylinder):
        radius = shape.width / 2
        height = shape.height / 2
        return f'{prefix}<geom name="{name}" type="cylinder" size="{radius:.3f} {height:.3f}" rgba="{rgba}"/>'
    else:
        # Default to sphere
        return f'{prefix}<geom name="{name}" type="sphere" size="0.1" rgba="{rgba}"/>'


def create_wind_turbine_world():
    """Create the wind turbine world without ROS dependencies."""
    world = World()
    with world.modify_world():
        # Root Body
        root = Body(name=PrefixedName("root"))

        # Degrees of freedom
        rotor_blade_dof = DegreeOfFreedom(name=PrefixedName('rotor_blade'))
        rotor_blade_dof.limits.upper.position = 0
        rotor_blade_dof.limits.lower.position = 1.606
        world.add_degree_of_freedom(rotor_blade_dof)

        wind_speed = DegreeOfFreedom(name=PrefixedName('wind_speed'))
        world.add_degree_of_freedom(wind_speed)

        # Create wind turbines
        wind2_hub = WindTurbine.create_with_new_body_in_world(
            world=world,
            parent=root,
            name=PrefixedName("wind2"),
            tower_height=3.0,
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=5, y=5, z=0.1)
        )

        wind3_hub = WindTurbine.create_with_new_body_in_world(
            world=world,
            parent=root,
            name=PrefixedName("wind3"),
            tower_height=10.0,
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=-5, y=5, z=0.1, yaw=np.pi/2)
        )

    return world


def load_and_run():
    """Load the wind turbine model and run it in MuJoCo viewer."""
    print("Creating wind turbine semantic digital twin...")
    world = create_wind_turbine_world()

    print("Converting to MuJoCo XML...")
    xml_string = generate_mujoco_xml(world)

    # Save XML for inspection
    xml_path = "wind_farm.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_string)
    print(f"Saved MuJoCo XML to {xml_path}")

    # Load in MuJoCo
    print("Loading in MuJoCo...")
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Launch viewer
    print("Launching MuJoCo viewer...")
    print("Press ESC to exit the viewer")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    load_and_run()
