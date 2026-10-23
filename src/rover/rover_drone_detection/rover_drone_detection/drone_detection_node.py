#!/usr/bin/env python3
"""Waiter-FPV-drone detection on one camera: find the drones, draw them.

    image                    sensor_msgs/Image, in (remapped to a rover_cameras topic)
    drones/detections        vision_msgs/Detection2DArray, every processed frame
    drones/image_annotated   sensor_msgs/Image, boxes drawn; only drawn with subscribers

    ros2 launch rover_drone_detection drone_detection.launch.py
    ros2 launch rover_drone_detection drone_detection.launch.py model:=drone_yolo26n_640.pt

Never drives anything: it only reports. Inference runs on a timer at `rate`
on the newest frame only, so a slow model drops frames instead of falling
behind.
"""

import array
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from rover_drone_detection.detector import Detector, resolve_model_path
from rover_drone_detection.draw import annotate
from rover_drone_detection.image_msg import bgr_fields, to_bgr


class DroneDetectionNode(Node):

    def __init__(self):
        super().__init__('drone_detection_node')
        ro = ParameterDescriptor(read_only=True)
        p = lambda name, default: self.declare_parameter(name, default, ro).value  # noqa: E731
        model = str(p('model', 'drone_yolo11s_640.pt'))
        conf = float(p('conf', 0.25))
        iou = float(p('iou', 0.5))
        imgsz = int(p('imgsz', 640))
        device = str(p('device', ''))
        half = bool(p('half', False))
        rate = float(p('rate', 10.0))

        models_dir = os.path.join(get_package_share_directory('rover_drone_detection'), 'models')
        path = resolve_model_path(model, models_dir)
        self.get_logger().info(f'loading {path}')
        self.detector = Detector(path, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half)
        t0 = time.monotonic()
        self.detector.warmup()
        self.get_logger().info(
            f'model ready ({(time.monotonic() - t0) * 1e3:.0f} ms warmup), classes '
            f'{self.detector.names}')

        self._frame = None
        self._in_view = False
        self._frames = 0
        self._infer_ms = 0.0

        self.det_pub = self.create_publisher(Detection2DArray, 'drones/detections', 10)
        self.image_pub = self.create_publisher(Image, 'drones/image_annotated',
                                               qos_profile_sensor_data)
        # Depth 1 best effort: only the newest frame matters.
        self.create_subscription(Image, 'image', self._on_image, qos_profile_sensor_data)
        self.create_timer(1.0 / max(rate, 0.1), self._tick)
        self.create_timer(10.0, self._report)

        self.get_logger().info(
            f'drone_detection_node: {self.resolve_topic_name("image")} at <= {rate:g} Hz')

    def _on_image(self, msg):
        self._frame = msg

    def _tick(self):
        msg, self._frame = self._frame, None
        if msg is None:
            return
        try:
            bgr = to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(f'cannot convert {msg.encoding} image: {exc}',
                                   throttle_duration_sec=5.0)
            return

        t0 = time.monotonic()
        dets = self.detector.detect(bgr)
        self._infer_ms = 0.9 * self._infer_ms + 0.1 * (time.monotonic() - t0) * 1e3
        self._frames += 1

        self.det_pub.publish(self._detections_msg(msg.header, dets))

        if self.image_pub.get_subscription_count() > 0:
            out = Image(header=msg.header, encoding='bgr8')
            out.height, out.width, out.step, data = bgr_fields(annotate(bgr, dets))
            out.data = array.array('B', data)
            self.image_pub.publish(out)

        if bool(dets) != self._in_view:
            self._in_view = bool(dets)
            if dets:
                self.get_logger().warn(f'{len(dets)} drone(s) in view, best score '
                                       f'{max(d.score for d in dets):.2f}')
            else:
                self.get_logger().info('no drone in view')

    def _report(self):
        if self._frames:
            self.get_logger().info(
                f'{self._frames / 10.0:.1f} fps processed, inference {self._infer_ms:.0f} ms')
        self._frames = 0

    def _detections_msg(self, header, dets):
        out = Detection2DArray(header=header)
        for d in dets:
            det = Detection2D(header=header)
            det.bbox.center.position.x = d.cx
            det.bbox.center.position.y = d.cy
            det.bbox.size_x = d.width
            det.bbox.size_y = d.height
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.label
            hyp.hypothesis.score = d.score
            det.results.append(hyp)
            out.detections.append(det)
        return out


def main():
    rclpy.init()
    node = DroneDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
