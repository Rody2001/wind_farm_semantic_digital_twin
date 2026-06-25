"""
3D model of Wind Turbine using Semantic Digital Twin framework.
"""

import threading, time, sys
from dataclasses import dataclass, field
from time import sleep

import numpy as np
import rclpy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher

from semantic_annotations import Tower, Nacelle, RotorBlades, Hub, TowerBase
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, ActiveConnection, ActiveConnection1DOF, \
    HasUpdateState, RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import VelocityVariable, DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

import semantic_digital_twin.spatial_types.spatial_types as cas

from wind_farm_export import combined_specs


@dataclass
class WindTurbine(ActiveConnection1DOF, HasUpdateState):
    rotor_dof: DegreeOfFreedom = field(default=None, kw_only=True)
    wind_dof: DegreeOfFreedom = field(default=None, kw_only=True)
    rotor_blade_angle_dof: DegreeOfFreedom = field(default=None, kw_only=True)

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
        """
        Create a complete wind turbine structure with 3 rotor blades.

        Args:
            world: The World instance
            parent: Parent body to attach the turbine to
            name: Base name for the turbine components
            tower_height: Height of the tower
            tower_width: Width/thickness of the tower
            nacelle_length: Length of the nacelle
            rotor_blade_length: Length of each rotor blade
            base_size: Size of the tower base
            parent_T_connection: Transform from parent to turbine base
        """
        if parent_T_connection is None:
            parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0.2)

        if rotor_blade_length == 0:
            rotor_blade_length = tower_height*(8475/15797)

        elements_conns = []
        elements_annotations = []

        nacelle_length = tower_height * (1500/15797)

        blade_x = tower_height * (0.04 / 5)
        blade_y = tower_height * (0.18/5)

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
        blade1_box = Box(scale=Scale(blade_x, blade_y, rotor_blade_length), color=white)
        blade1_body = Body(name=PrefixedName(f"{name}_rotor_blade1"),
                          visual=ShapeCollection([blade1_box]),
                          collision=ShapeCollection([blade1_box]))
        blade1_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=hub_body, child=blade1_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=-4.5*tower_width, z=2.5*tower_width, roll=-np.pi*2/3),  #x=-0.05, y=-(rotor_blade_length)/2, z=0.45, roll=-np.pi*2/3),
            axis=cas.Vector3.Z())
        blade1_annotation = RotorBlades(root=blade1_body, name=PrefixedName(f"{name}_rotor_blade1"))
        elements_conns.append(blade1_conn)
        elements_annotations.append(blade1_annotation)

        # Blade 2
        blade2_box = Box(scale=Scale(blade_x, blade_y, rotor_blade_length), color=white)
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
        blade3_box = Box(scale=Scale(blade_x, blade_y, rotor_blade_length), color=white)
        blade3_body = Body(name=PrefixedName(f"{name}_rotor_blade3"),
                          visual=ShapeCollection([blade3_box]),
                          collision=ShapeCollection([blade3_box]))
        blade3_conn = RevoluteConnection.create_with_dofs(
            world=world, parent=hub_body, child=blade3_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-(rotor_blade_length/2) - (nacelle_height/2)), # x=-0.05, y=0.0, z=-(rotor_blade_length)/2)
            axis=cas.Vector3.Z())
        blade3_annotation = RotorBlades(root=blade3_body, name=PrefixedName(f"{name}_rotor_blade3"))
        elements_conns.append(blade3_conn)
        elements_annotations.append(blade3_annotation)

        # Add all to world
        for conn in elements_conns:
            world.add_connection(conn)
        for annotation in elements_annotations:
            world.add_semantic_annotation(annotation)

        return hub_conn  # Return hub connection to access DOF

    def add_to_world(self, world: World):

        self._connection_T_child_expression = (
            cas.HomogeneousTransformationMatrix.from_xyz_axis_angle(
                axis=self.axis,
                angle=0,
                child_frame=self.child,
            )
        )

    def update_state(self, dt: float):
        wind_vel = self._world.state[self.wind_dof.id].velocity
        rotor_angle = self._world.state[self.rotor_dof.id].position

        self.rotor_dof.position = self.rotor_dof.position + dt
        self._world.state[self.rotor_dof.id].velocity = wind_vel * rotor_angle


def main():
    """
    Entry point for constructing and visualizing the Wind Turbine.

    This function programmatically defines a hierarchical wind turbine model consisting of:
        - Tower base
        - Tower shaft
        - Nacelle
        - Three rotor blades
        - Central hub

    Each component is represented as a Body with associated geometric primitives, semantic
    annotations, and parent–child spatial relationships expressed through FixedConnections.

    The assembled world model is published to ROS 2 using visualization markers so that the
    full turbine structure can be inspected in RViz2.

    Visualization:
        Launch RViz2 and add a "Marker" display subscribed to:
            /viz_marker

    Purpose:
        Enables testing, debugging, and validation of semantic digital twin workflows,
        world modeling pipelines, and visualization toolchains for robotics and simulation.
    """


    world = World()
    with world.modify_world():

        # ----- Define colors -----
        red = Color(1, 0, 0)
        white = Color(1, 1, 1)
        blue = Color(0, 0, 1)
        brown = Color(1, 0.5, 0.25)
        green = Color(0, 1, 0)

        # ----- Root Body -----
        root = Body(name=PrefixedName("root"))


        rotor_blade_dof = DegreeOfFreedom(name=PrefixedName('rotor_blade'))
        rotor_blade_dof.limits.upper.position = 0
        rotor_blade_dof.limits.lower.position = 1.606
        world.add_degree_of_freedom(rotor_blade_dof)

        # commented out, because the world doesn't like DoFs that are not linked to a connection
        wind_speed = DegreeOfFreedom(name=PrefixedName('wind_speed'))
        world.add_degree_of_freedom(wind_speed)

        # =====================================================================
        # wind turbines  (built from ALL farms via combined_specs - one source of truth)
        # =====================================================================
        turbine_hubs = {}
        for spec in combined_specs():
            hub = WindTurbine.create_with_new_body_in_world(
                world=world,
                parent=root,
                name=PrefixedName(spec.name),
                tower_height=spec.tower_height,
                rotor_blade_length=spec.rotor_blade_length,
                parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=spec.x, y=spec.y, z=spec.z, yaw=spec.yaw),
            )
            turbine_hubs[spec.name] = hub


    # =====================================================================
    # ROS2 Node and Visualization Publisher
    # =====================================================================
    rclpy.init()
    node = rclpy.create_node("semantic_digital_twin")
    viz = VizMarkerPublisher(_world=world, node=node)
    viz.with_tf_publisher()
    # Spin ROS2 node in background thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    return world

#main()
#print(main().semantic_annotations)
if __name__ == "__main__":
        main()
