#!/usr/bin/env python3
"""Road keeping between the operator and twist_mux.

Takes the operator's command on cmd_vel_in (cmd_vel_remote_raw) and the
latest road/state, and publishes the command twist_mux sees on cmd_vel_out
(cmd_vel_remote) — the only publisher there. Both names are set in
config/road_assist.yaml. One command out per command in, never on a timer, so
this node can never keep the rover driving after the operator stops. Only
angular.z is ever changed.

    ros2 run rover_road_assist road_assist_node
    ros2 service call /rover/road_assist/toggle std_srvs/srv/Trigger

Correction is switched on and off at runtime, by road_assist/toggle or by
the `assist` parameter. Whether the status says WARN rather than OFF while it
is off is the read-only `warn` setting in config/road_assist.yaml.

The maths lives in steering_law and assist_state; this file is only the ROS
side of it.
"""

from dataclasses import fields

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger

from indomitus_interfaces.msg import RoadDeviation

from rover_road_assist.assist_state import AssistParams, Mode, RoadAssist, RoadEstimate
from rover_road_assist.steering_law import SteeringParams, road_error

# Latched, so a UI that starts later still sees the current status.
STATUS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RoadAssistNode(Node):

    def __init__(self):
        super().__init__('road_assist_node')

        self.declare_parameter('assist', False, ParameterDescriptor(
            description='Correct the steering; the only parameter that can change at runtime'))
        read_only = ParameterDescriptor(read_only=True)
        self.declare_parameter('warn', False, read_only)
        self.declare_parameter('status_rate', 5.0, read_only)
        self.declare_parameter('cmd_vel_in', 'cmd_vel_remote_raw', read_only)
        self.declare_parameter('cmd_vel_out', 'cmd_vel_remote', read_only)
        self._steering = self._declare_params(SteeringParams)
        self._assist = RoadAssist(self._steering, self._declare_params(AssistParams))
        self._status = None
        self.add_on_set_parameters_callback(self._on_set_parameters)

        cmd_vel_in = str(self.get_parameter('cmd_vel_in').value)
        cmd_vel_out = str(self.get_parameter('cmd_vel_out').value)
        self._pub = self.create_publisher(Twist, cmd_vel_out, 10)
        self._status_pub = self.create_publisher(String, 'road_assist/status', STATUS_QOS)
        self.create_service(Trigger, 'road_assist/toggle', self._on_toggle)
        self.create_subscription(Twist, cmd_vel_in, self._on_command, 10)
        # Best effort matches a camera pipeline publishing either way.
        self.create_subscription(RoadDeviation, 'road/state', self._on_road,
                                 qos_profile_sensor_data)
        # The status also has to go STALE while nobody is driving.
        status_rate = float(self.get_parameter('status_rate').value)
        self.create_timer(1.0 / max(status_rate, 1.0), self._publish_status)

        self.get_logger().info(
            f'road_assist_node: {cmd_vel_in} -> {cmd_vel_out}, {self._mode().value}')

    def _declare_params(self, cls):
        """One read-only parameter per field of cls, defaulting to the field's."""
        read_only = ParameterDescriptor(read_only=True)
        return cls(**{
            f.name: float(self.declare_parameter(f.name, f.default, read_only).value)
            for f in fields(cls)
        })

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Only logs: the declared bool type already refuses anything else.
        for param in params:
            if param.name == 'assist':
                self.get_logger().info(
                    f'road_assist_node: assist {"on" if param.value else "off"}')
        return SetParametersResult(successful=True)

    def _on_toggle(self, _request, response):
        # Through the parameter, so `ros2 param get` never disagrees with it.
        enable = not self.get_parameter('assist').value
        result = self.set_parameters([Parameter('assist', Parameter.Type.BOOL, enable)])[0]
        response.success = result.successful
        response.message = (f'assist {"on" if enable else "off"}'
                            if result.successful else result.reason)
        return response

    def _mode(self) -> Mode:
        if self.get_parameter('assist').value:
            return Mode.ASSIST
        return Mode.WARN if self.get_parameter('warn').value else Mode.OFF

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_road(self, msg: RoadDeviation):
        offset, heading = road_error(
            msg.lateral_offset, msg.heading_error, self._steering.road_width)
        self._assist.on_road(RoadEstimate(
            offset=offset,
            heading=heading,
            confidence=msg.confidence,
            valid=msg.valid,
            stamp=Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
        ), self._now())

    def _on_command(self, msg: Twist):
        msg.angular.z = self._assist.on_command(
            msg.linear.x, msg.angular.z, self._now(), self._mode())
        self._pub.publish(msg)
        self._publish_status()

    def _publish_status(self):
        status = self._assist.status(self._now(), self._mode())
        if status != self._status:
            self._status = status
            self._status_pub.publish(String(data=status.value))


def main():
    rclpy.init()
    node = RoadAssistNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
