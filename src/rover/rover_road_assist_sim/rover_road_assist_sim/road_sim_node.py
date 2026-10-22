#!/usr/bin/env python3
"""Fake road perception for simulation: road/state from the true pose.

Reads the rover's ground-truth pose from Gazebo, measures it against the
configured track, and publishes RoadDeviation at camera rate — what
road_assist_node would get from the real camera pipeline, with optional
noise, latency and a modelled confidence. Also draws the track and the
rover's true trail in RViz.

    ros2 launch rover_road_assist_sim road_assist_sim.launch.py

The maths lives in track and perception_model; this file is only the ROS
side of it.
"""

import math
from dataclasses import fields

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from indomitus_interfaces.msg import RoadDeviation

from rover_road_assist_sim.perception_model import (
    CameraParams, ConfidenceParams, DelayLine, NoiseParams, RoadCamera,
)
from rover_road_assist_sim.track import Pose, build_track

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

#: [m] Spacing of the drawn centre line and edges along arcs.
MARKER_STEP = 0.5
#: [m] The trail only grows once the rover has moved this far.
TRAIL_STEP = 0.1
TRAIL_POINTS = 5000


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _color(r, g, b, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


class RoadSimNode(Node):

    def __init__(self):
        super().__init__('road_sim_node')
        read_only = ParameterDescriptor(read_only=True)

        def param(name, default):
            return self.declare_parameter(name, default, read_only).value

        ground_truth_topic = param('ground_truth_topic', 'ground_truth/odom')
        road_state_topic = param('road_state_topic', 'road/state')
        markers_topic = param('markers_topic', 'road/markers')
        self._frame_id = param('frame_id', 'odom')
        publish_rate = float(param('publish_rate', 15.0))

        self._track = build_track(
            param('track.start', [-5.0, 0.0, 0.0]),
            param('track.scale', 1.0),
            param('track.main', ['straight 50']),
            param('track.branch.from', -1),
            param('track.branch.segments', ['straight 1']),
        )
        self._camera_params = self._declare_params('', CameraParams)
        noise = self._declare_params('noise.', NoiseParams)
        latency_enabled = bool(param('latency.enabled', False))
        latency = float(param('latency.delay', 0.12)) if latency_enabled else 0.0
        confidence = self._declare_params('confidence.', ConfidenceParams)
        self._camera = RoadCamera(self._track, self._camera_params, noise, confidence)

        self._poses = DelayLine(latency)
        self._following = None
        self._trail = []

        self._pub = self.create_publisher(RoadDeviation, road_state_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, markers_topic, LATCHED)
        self._trail_pub = self.create_publisher(Marker, f'{markers_topic}/trail', 10)
        self.create_subscription(Odometry, ground_truth_topic, self._on_odom, 10)
        self.create_timer(1.0 / max(publish_rate, 1.0), self._tick)

        self._markers_pub.publish(self._road_markers())
        self.get_logger().info(
            f'road_sim_node: {ground_truth_topic} -> {road_state_topic}, '
            f'main {self._track.main.length:.0f} m'
            + (f', branch {self._track.branch.length:.0f} m' if self._track.branch else '')
            + f', noise {"on" if noise.enabled else "off"}, latency {latency:.2f} s, '
            f'confidence {confidence.mode}')

    def _declare_params(self, prefix: str, cls):
        """One read-only parameter per field of cls, defaulting to the field's."""
        read_only = ParameterDescriptor(read_only=True)
        return cls(**{
            f.name: type(f.default)(
                self.declare_parameter(prefix + f.name, f.default, read_only).value)
            for f in fields(cls)
        })

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self._poses.push(stamp, Pose(p.position.x, p.position.y, _yaw(p.orientation)))

    def _tick(self):
        sample = self._poses.at(self._now())
        if sample is None:
            return
        stamp, pose = sample
        seen = self._camera.observe(pose)

        msg = RoadDeviation()
        msg.header.stamp = Time(nanoseconds=int(stamp * 1e9)).to_msg()
        msg.header.frame_id = 'base_link'
        msg.lateral_offset = float(seen.lateral_offset)
        msg.heading_error = float(seen.heading_error)
        msg.confidence = float(seen.confidence)
        msg.valid = seen.valid
        self._pub.publish(msg)

        if seen.path != self._following:
            self._following = seen.path
            self.get_logger().info(f'following: {seen.path or "no road in view"}')
        self._publish_trail(pose)

    # ── RViz ────────────────────────────────────────────────────────────────

    def _line(self, ns: str, ident: int, points, width: float, color: ColorRGBA) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns, m.id = ns, ident
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color = color
        m.points = [Point(x=x, y=y, z=0.02) for x, y in points]
        return m

    def _road_markers(self) -> MarkerArray:
        half = self._camera_params.road_width / 2.0
        out = MarkerArray()
        for ident, path in enumerate(self._track.paths):
            poses = [path.pose_at(s) for s in path.samples(MARKER_STEP)]
            out.markers.append(self._line('centre', ident, [(q.x, q.y) for q in poses],
                                          0.08, _color(0.95, 0.75, 0.1)))
            for side, offset in ((0, half), (1, -half)):
                out.markers.append(self._line(
                    'edge', 2 * ident + side, [q.left(offset) for q in poses],
                    0.06, _color(0.9, 0.9, 0.9)))
        return out

    def _publish_trail(self, pose: Pose):
        last = self._trail[-1] if self._trail else None
        if last and (pose.x - last[0]) ** 2 + (pose.y - last[1]) ** 2 < TRAIL_STEP ** 2:
            return
        self._trail.append((pose.x, pose.y))
        del self._trail[:-TRAIL_POINTS]
        self._trail_pub.publish(self._line('trail', 0, self._trail, 0.05,
                                           _color(0.2, 0.6, 1.0)))


def main():
    rclpy.init()
    node = RoadSimNode()
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
