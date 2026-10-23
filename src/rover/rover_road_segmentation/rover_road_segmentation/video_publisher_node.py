#!/usr/bin/env python3
"""Publishes frames from an existing video file to a camera Image topic.

Useful for replaying dataset videos (e.g., dev_0001.mp4 ... dev_0012.mp4)
into the perception pipeline as if coming from a real hardware camera.

Publishes:
    <image_topic> (sensor_msgs/msg/Image) with current ROS wall timestamp.
"""
from pathlib import Path

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from rover_road_segmentation import config as C

CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def resolve_video_path(path_str):
    p = Path(path_str).expanduser()
    if p.exists():
        return p
    data_dir = C.find_data_dir()
    if data_dir is not None:
        candidates = [
            data_dir / p.name,
            data_dir / 'videos' / p.name,
        ]
        if len(p.parts) > 1:
            candidates.append(data_dir / Path(*p.parts[1:]))
        for cand in candidates:
            if cand.exists():
                return cand
    return p


class VideoPublisherNode(Node):

    def __init__(self):
        super().__init__('video_publisher_node')
        p = self.declare_parameter

        default_video = ''
        data_dir = C.find_data_dir()
        if data_dir and (data_dir / 'videos' / 'dev_0001.mp4').exists():
            default_video = str(data_dir / 'videos' / 'dev_0001.mp4')

        video_path_raw = p('video_path', default_video).value
        self.image_topic = p('image_topic', '/rover/cameras/arducam_front_bottom/image_raw').value
        self.frame_id = p('frame_id', 'arducam_front_bottom_optical_frame').value
        target_fps = float(p('fps', 0.0).value)
        self.loop = bool(p('loop', True).value)

        self.video_path = resolve_video_path(video_path_raw)
        if not self.video_path.exists():
            self.get_logger().error(f"Video file not found: {self.video_path} (from {video_path_raw})")
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = target_fps if target_fps > 0 else video_fps
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.pub = self.create_publisher(Image, self.image_topic, CAMERA_QOS)
        timer_period = 1.0 / max(1.0, self.fps)
        self.timer = self.create_timer(timer_period, self._on_timer)

        self.get_logger().info(
            f"video_publisher_node: {self.video_path.name} ({self.w}x{self.h} @ {self.fps:.1f} FPS, {self.total_frames} frames) "
            f"-> {self.image_topic} (loop={self.loop})")

    def _on_timer(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warn("Reached end of video stream.")
                return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()

        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = VideoPublisherNode()
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
