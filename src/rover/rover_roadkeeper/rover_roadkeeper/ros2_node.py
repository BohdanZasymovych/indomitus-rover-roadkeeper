#!/usr/bin/env python3
"""
road_offset_perception_node - thin ROS 2 wrapper around RoadPerceptionCore.

  Sub:  <image_topic>        sensor_msgs/Image           (default /rover/front_camera/image_raw)
  Pub:  <offset_topic>       geometry_msgs/Vector3Stamped (default /rover/road_offset)
            vector.x = smoothed offset [-1, 1]   (<0 road centre is left, >0 right)
            vector.y = road confidence [0, 1]
            vector.z = 1.0 if valid road measurement this frame else 0.0
        <debug_image_topic>  sensor_msgs/Image (bgr8), only if publish_debug_image:=true

This node ONLY does perception. Steering / speed control belongs to a downstream node.
Images are decoded straight from the message buffer (no cv_bridge -> no C++ ABI issues).
All rclpy imports are confined to this file.

    ros2 run rover_roadkeeper road_offset_node --ros-args -p publish_debug_image:=true
"""
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image

from rover_roadkeeper import config as C
from rover_roadkeeper.perception.pipeline import RoadPerceptionCore

import cv2


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Decode common sensor_msgs/Image encodings into a BGR uint8 array."""
    enc = msg.encoding.lower()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1, "8uc3": 3}.get(enc)
    if channels is None:
        raise ValueError(f"Unsupported image encoding '{msg.encoding}'")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = buf.reshape(msg.height, msg.step)[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if channels == 1:
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)
    return img


def resolve_weights(param_value: str) -> str:
    if param_value:
        return param_value
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory("rover_roadkeeper")) / "weights" / C.DEFAULT_CHECKPOINT.name
        if installed.exists():
            return str(installed)
    except Exception:  # package not installed via colcon
        pass
    return str(C.DEFAULT_CHECKPOINT)


def bgr_to_image_msg(img: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


class RoadOffsetPerceptionNode(Node):
    def __init__(self):
        super().__init__(C.ROS_NODE_NAME)

        self.declare_parameter("image_topic", C.ROS_IMAGE_TOPIC)
        self.declare_parameter("offset_topic", C.ROS_OFFSET_TOPIC)
        self.declare_parameter("debug_image_topic", C.ROS_DEBUG_IMAGE_TOPIC)
        self.declare_parameter("weights_path", "")       # "" -> installed share dir / source tree
        self.declare_parameter("prob_threshold", C.ROAD_PROB_THRESHOLD)
        self.declare_parameter("device", "")              # "" -> auto (cuda if available)
        self.declare_parameter("publish_debug_image", False)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.publish_debug = bool(p("publish_debug_image"))

        weights = resolve_weights(p("weights_path"))
        self.get_logger().info(f"Loading road segmenter from {weights} ...")
        self.core = RoadPerceptionCore(
            checkpoint_path=weights,
            device=p("device") or None,
            prob_threshold=float(p("prob_threshold")),
        )
        self.get_logger().info(f"Model ready on {self.core.device}")

        # depth=1 best-effort: always process the newest frame, drop stale ones.
        self.sub = self.create_subscription(Image, p("image_topic"), self.on_image, qos_profile_sensor_data)
        reliable = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_offset = self.create_publisher(Vector3Stamped, p("offset_topic"), reliable)
        self.pub_debug = (self.create_publisher(Image, p("debug_image_topic"), qos_profile_sensor_data)
                          if self.publish_debug else None)

        self.get_logger().info(f"Listening on {p('image_topic')} -> publishing {p('offset_topic')}")

    def on_image(self, msg: Image) -> None:
        try:
            frame = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return

        want_debug = self.pub_debug is not None and self.pub_debug.get_subscription_count() > 0
        r = self.core.process_frame(frame, render_debug=want_debug)

        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = float(r.offset)
        out.vector.y = float(r.road_confidence)
        out.vector.z = 1.0 if r.is_valid else 0.0
        self.pub_offset.publish(out)

        if want_debug:
            self.pub_debug.publish(bgr_to_image_msg(r.debug_frame, msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = RoadOffsetPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
