#!/usr/bin/env python3
"""Visualizes the camera image, road segmentation mask, and RoadDeviation telemetry.

Subscribes to:
    - Camera Image: /rover/cameras/arducam_front_bottom/image_raw
    - Road Mask:    /rover/perception/road_mask (mono8, 0/1)
    - Deviation:    /rover/road/state (indomitus_interfaces/msg/RoadDeviation)

Renders:
    - Live camera feed with translucent road overlay
    - Scanlines and detected road center points
    - Rover centerline
    - Real-time gauge for lateral offset, heading error, confidence, and FPS
Publishes:
    - ~/dashboard (sensor_msgs/msg/Image)
    - Optionally displays via cv2.imshow when show:=true
    - Optionally records to an mp4 file when save_output:=true
"""
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from indomitus_interfaces.msg import RoadDeviation
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from rover_road_segmentation.dashboard_visualizer import render_dashboard
from rover_road_segmentation.image_codec import image_to_array


class RoadDeviationVisualizerNode(Node):

    def __init__(self):
        super().__init__('road_deviation_visualizer_node')
        p = self.declare_parameter

        image_topic = p('image_topic', '/rover/cameras/arducam_front_bottom/image_raw').value
        mask_topic = p('mask_topic', '/rover/perception/road_mask').value
        deviation_topic = p('deviation_topic', '/rover/road/state').value
        self.show = bool(p('show', True).value)
        self.save_output = bool(p('save_output', False).value)
        self.output_path = p('output_path', 'outputs/demo_ros_pipeline.mp4').value
        self.alpha = float(p('alpha', 0.45).value)

        self._last_frame = None
        self._last_mask = None
        self._last_deviation = RoadDeviation()
        self._last_deviation.valid = False
        self._frames_count = 0
        self._last_time = time.monotonic()
        self._fps = 0.0

        self.writer = None
        self.pub_vis = self.create_publisher(Image, '~/dashboard', 1)

        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Image, mask_topic, self._on_mask, qos_profile_sensor_data)
        self.create_subscription(RoadDeviation, deviation_topic, self._on_deviation, qos_profile_sensor_data)

        self.get_logger().info(
            f"road_deviation_visualizer_node started.\n"
            f"  Subscribing to image:     {image_topic}\n"
            f"  Subscribing to mask:      {mask_topic}\n"
            f"  Subscribing to deviation: {deviation_topic}\n"
            f"  Show window: {self.show} | Save MP4: {self.save_output}")

    def _on_deviation(self, msg: RoadDeviation):
        self._last_deviation = msg

    def _on_mask(self, msg: Image):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        self._last_mask = data.reshape(msg.height, msg.step)[:, :msg.width]

    def _on_image(self, msg: Image):
        t0 = time.perf_counter()
        try:
            frame, order = image_to_array(msg)
            if order == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif order == 'gray':
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            self.get_logger().error(f"Failed to decode image: {exc}")
            return

        self._last_frame = frame
        h, w = frame.shape[:2]

        mask = self._last_mask
        if mask is None or mask.shape != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)

        # FPS calculation
        self._frames_count += 1
        now = time.monotonic()
        if now - self._last_time >= 1.0:
            self._fps = self._frames_count / (now - self._last_time)
            self._frames_count = 0
            self._last_time = now

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Render composite dashboard
        dashboard = render_dashboard(
            frame_bgr=frame,
            mask=mask,
            deviation=self._last_deviation,
            fps=self._fps,
            latency_ms=latency_ms,
            alpha=self.alpha,
        )

        # Publish dashboard image
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = dashboard.shape[:2]
        out_msg.encoding = 'bgr8'
        out_msg.is_bigendian = 0
        out_msg.step = dashboard.shape[1] * 3
        out_msg.data = dashboard.tobytes()
        self.pub_vis.publish(out_msg)

        # Save to video if enabled
        if self.save_output:
            if self.writer is None:
                p = Path(self.output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(
                    str(p),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    max(10.0, self._fps or 20.0),
                    (w, h),
                )
            self.writer.write(dashboard)

        # Display GUI window if enabled
        if self.show:
            cv2.imshow("Road Assist Demo Dashboard", dashboard)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = RoadDeviationVisualizerNode()
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
