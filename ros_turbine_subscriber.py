"""
ros_turbine_subscriber.py
===================================================================
Subscribes to the per-turbine ROS2 topics published by wind_turbine_sim.py
(when launched with --publish): rpm, power_w, energy_kwh per turbine, plus
the two farm-wide totals. Stores the latest values so SemanticWindDriver's
queries return exactly what MuJoCo actually published -- no independent
re-computation, no drift between the two worlds.

Message type assumed: std_msgs/msg/Float64. Check with:
    ros2 topic type /wind_farm/Farm_East_1/rpm
If it reports Float32 instead, change the import below to
`from std_msgs.msg import Float32 as Float64` (or just swap the name).
===================================================================
"""

from dataclasses import dataclass
from typing import Dict, Iterable

from std_msgs.msg import Float64


@dataclass
class TurbinePublishedState:
    rpm: float = 0.0
    power_w: float = 0.0
    energy_kwh: float = 0.0
    rpm_seen: bool = False       # True once at least one rpm message has actually arrived


class RosTurbineSubscriber:
    """Subscribes to /wind_farm/<name>/{rpm,power_w,energy_kwh} for every turbine name given."""

    def __init__(self, node, turbine_names: Iterable[str], topic_prefix: str = "/wind_farm"):
        self.node = node
        self.published: Dict[str, TurbinePublishedState] = {
            name: TurbinePublishedState() for name in turbine_names
        }
        self.total_power_w = 0.0
        self.total_energy_kwh = 0.0
        self._subs = []   # keep references alive; rclpy drops subscriptions that get GC'd

        for name in self.published:
            self._subs.append(node.create_subscription(
                Float64, f"{topic_prefix}/{name}/rpm", self._make_cb(name, "rpm"), 10))
            self._subs.append(node.create_subscription(
                Float64, f"{topic_prefix}/{name}/power_w", self._make_cb(name, "power_w"), 10))
            self._subs.append(node.create_subscription(
                Float64, f"{topic_prefix}/{name}/energy_kwh", self._make_cb(name, "energy_kwh"), 10))

        self._subs.append(node.create_subscription(
            Float64, f"{topic_prefix}/total_power_w", self._make_total_cb("total_power_w"), 10))
        self._subs.append(node.create_subscription(
            Float64, f"{topic_prefix}/total_energy_kwh", self._make_total_cb("total_energy_kwh"), 10))

    def _make_cb(self, name: str, field_name: str):
        def _cb(msg):
            state = self.published[name]
            setattr(state, field_name, msg.data)
            if field_name == "rpm":
                state.rpm_seen = True
        return _cb

    def _make_total_cb(self, attr_name: str):
        def _cb(msg):
            setattr(self, attr_name, msg.data)
        return _cb

    def get(self, name: str) -> TurbinePublishedState:
        return self.published[name]
