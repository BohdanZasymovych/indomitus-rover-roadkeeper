#!/usr/bin/env python3
"""Turns the road mask into a RoadDeviation.

Subscribes to a mono8 mask (nonzero = road) from the segmentation node and
publishes how far the vehicle is from the road centre and how the road bends.
The stamp and frame_id of the mask are passed through so a consumer can tell how
old the measurement is. See road_deviation.py for the maths.
"""

import numpy as np
import rclpy
from indomitus_interfaces.msg import RoadDeviation
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image

from rover_road_assist.road_deviation import DeviationFilter, estimate_deviation

MONO8_ENCODINGS = ('mono8', '8UC1')


def mask_from_image(msg):
    """The 2-D uint8 array of a mono8 Image, without cv_bridge."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    return data.reshape(msg.height, msg.step)[:, :msg.width]


class RoadDeviationNode(Node):

    def __init__(self):
        super().__init__('road_deviation_node')

        self.declare_parameter('mask_topic', '/rover/perception/road_mask')
        self.declare_parameter('deviation_topic', '/rover/road/state')
        self.declare_parameter('row_fractions', [0.6, 0.7, 0.8, 0.9])
        self.declare_parameter('horizon_fraction', 0.4)
        # Moves the reference column for a camera not mounted on the vehicle axis.
        self.declare_parameter('target_col_fraction', 0.5)
        self.declare_parameter('min_run_fraction', 0.05)
        self.declare_parameter('min_rows', 2)
        self.declare_parameter('min_road_fraction', 0.02)
        self.declare_parameter('full_confidence_road_fraction', 0.2)
        self.declare_parameter('smoothing_alpha', 0.5)

        self.filter = DeviationFilter(self.get_parameter('smoothing_alpha').value)
        self._warned_encoding = False

        mask_topic = self.get_parameter('mask_topic').value
        deviation_topic = self.get_parameter('deviation_topic').value
        self.pub = self.create_publisher(RoadDeviation, deviation_topic, QoSProfile(depth=1))
        self.create_subscription(Image, mask_topic, self._on_mask, qos_profile_sensor_data)

        self.get_logger().info(f'road_deviation_node: {mask_topic} -> {deviation_topic}')

    def _on_mask(self, msg):
        if msg.encoding not in MONO8_ENCODINGS:
            if not self._warned_encoding:
                self._warned_encoding = True
                self.get_logger().error(
                    f'mask encoding {msg.encoding!r} is not mono8, dropping frames')
            return

        p = self.get_parameter
        deviation = estimate_deviation(
            mask_from_image(msg),
            row_fractions=p('row_fractions').value,
            horizon_fraction=p('horizon_fraction').value,
            target_col_fraction=p('target_col_fraction').value,
            min_run_fraction=p('min_run_fraction').value,
            min_rows=p('min_rows').value,
            min_road_fraction=p('min_road_fraction').value,
            full_confidence_road_fraction=p('full_confidence_road_fraction').value,
        )
        deviation = self.filter.update(deviation)

        out = RoadDeviation()
        out.header = msg.header
        out.lateral_offset = deviation.lateral_offset
        out.heading_error = deviation.heading_error
        out.confidence = deviation.confidence
        out.valid = deviation.valid
        self.pub.publish(out)


def main():
    rclpy.init()
    node = RoadDeviationNode()
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
