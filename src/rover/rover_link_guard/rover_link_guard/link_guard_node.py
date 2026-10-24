#!/usr/bin/env python3
"""Drives the rover by itself while the operator's link is down.

Watches the operator's command on cmd_vel_in (cmd_vel_remote_raw); when it
goes quiet, publishes the fallback's command on cmd_vel_out (cmd_vel_ext),
twist_mux's lowest-priority input. So the operator takes the rover back by
simply driving again, with no handover logic here, and publishing stops the
moment the link is judged restored. Both names are set in config/link_guard.yaml.

While the link is up this only records the trail from odometry. The decisions
are in link_monitor and fallback; this file is only the ROS side of them.

    ros2 launch rover_link_guard link_guard.launch.py
    ros2 topic echo /rover/link_guard/status
"""

import math
from dataclasses import fields

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import String

from indomitus_interfaces.msg import RoadDeviation
from rover_road_assist.assist_state import RoadEstimate
from rover_road_assist.steering_law import SteeringParams, road_error

from rover_link_guard.fallback import Fallback, FallbackParams, Phase, Pose, RetreatMode
from rover_link_guard.link_monitor import LinkMonitor, LinkParams, LinkState, SpeedMemory

# Latched, so a UI that starts later still sees the current status.
STATUS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LinkGuardNode(Node):

    def __init__(self):
        super().__init__('link_guard_node')

        read_only = ParameterDescriptor(read_only=True)
        self.declare_parameter('cmd_vel_in', 'cmd_vel_remote_raw', read_only)
        self.declare_parameter('cmd_vel_out', 'cmd_vel_ext', read_only)
        self.declare_parameter('odom_topic', 'odom', read_only)
        self.declare_parameter('rate', 20.0, read_only)
        self.declare_parameter('retreat', RetreatMode.REVERSE.value, read_only)

        self._steering = self._declare_params(SteeringParams)
        link_params = self._declare_params(LinkParams)
        self._monitor = LinkMonitor(link_params)
        self._speed = SpeedMemory(link_params)
        retreat = RetreatMode(str(self.get_parameter('retreat').value))
        self._fallback = Fallback(
            self._steering, self._declare_params(FallbackParams), retreat)
        self._link = LinkState.OK
        self._status = None

        cmd_vel_in = str(self.get_parameter('cmd_vel_in').value)
        cmd_vel_out = str(self.get_parameter('cmd_vel_out').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self._pub = self.create_publisher(Twist, cmd_vel_out, 10)
        self._status_pub = self.create_publisher(String, 'link_guard/status', STATUS_QOS)
        self.create_subscription(Twist, cmd_vel_in, self._on_command, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        # Best effort matches a camera pipeline publishing either way.
        self.create_subscription(RoadDeviation, 'road/state', self._on_road,
                                 qos_profile_sensor_data)
        rate = float(self.get_parameter('rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self._publish_status()

        self.get_logger().info(
            f'link_guard_node: {cmd_vel_in} lost -> {cmd_vel_out}, retreat {retreat.value}')

    def _declare_params(self, cls):
        """One read-only parameter per field of cls, defaulting to the field's.

        A field two classes share (max_yaw_rate) is declared once, by whichever
        comes first, and both read that value: declaring it again would raise.
        """
        read_only = ParameterDescriptor(read_only=True)
        return cls(**{
            f.name: float(
                self.get_parameter(f.name).value if self.has_parameter(f.name)
                else self.declare_parameter(f.name, float(f.default), read_only).value)
            for f in fields(cls)
        })

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, msg: Twist):
        self._monitor.on_command(self._now())
        self._speed.on_command(self._now(), msg.linear.x)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self._fallback.on_pose(Pose(
            x=p.position.x, y=p.position.y, yaw=yaw_of(p.orientation),
            stamp=Time.from_msg(msg.header.stamp).nanoseconds * 1e-9))

    def _on_road(self, msg: RoadDeviation):
        offset, heading = road_error(
            msg.lateral_offset, msg.heading_error, self._steering.nominal_road_width)
        self._fallback.on_road(RoadEstimate(
            offset=offset, heading=heading, confidence=msg.confidence, valid=msg.valid,
            stamp=Time.from_msg(msg.header.stamp).nanoseconds * 1e-9))

    def _tick(self):
        now = self._now()
        phase = self._fallback.phase
        link = self._monitor.update(now)
        if link is not self._link:
            self._link = link
            if link is LinkState.LOST:
                v = self._speed.at_drop()
                self.get_logger().warn(f'link lost at {v:.2f} m/s, taking over')
                self._fallback.start(now, v)
            else:
                self.get_logger().info('link restored, handing back')
                self._fallback.stop()

        command = self._fallback.step(now)
        if command is not None:
            msg = Twist()
            msg.linear.x = command.v
            msg.angular.z = command.wz
            self._pub.publish(msg)
        if self._fallback.phase is not phase and self._fallback.phase is not Phase.IDLE:
            self.get_logger().info(
                f'link_guard: {self._fallback.phase.value} ({self._fallback.reason})')
        self._publish_status()

    def _publish_status(self):
        phase = self._fallback.phase
        status = 'LINK_OK' if phase is Phase.IDLE else phase.value
        if status != self._status:
            self._status = status
            self._status_pub.publish(String(data=status))


def main():
    rclpy.init()
    node = LinkGuardNode()
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
