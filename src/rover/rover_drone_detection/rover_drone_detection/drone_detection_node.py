#!/usr/bin/env python3
"""Waiter-FPV-drone detection on one camera: warn, draw, and log the position.

    image                    sensor_msgs/Image, in (remapped to a rover_cameras topic)
    camera_info              sensor_msgs/CameraInfo, in; K = 0 falls back to hfov_deg
    <gps_topic>              sensor_msgs/NavSatFix, in, optional (empty = off)
    drones/detections        vision_msgs/Detection2DArray, every processed frame
    drones/threat            indomitus_interfaces/DroneThreat, every processed frame
    drones/image_annotated   sensor_msgs/Image, boxes + banner; only drawn with subscribers

    ros2 launch rover_drone_detection drone_detection.launch.py
    ros2 launch rover_drone_detection drone_detection.launch.py model:=drone_yolo26n_640.pt

Never drives anything: it only reports. Inference runs on a timer at `rate`
on the newest frame only, so a slow model drops frames instead of falling
behind. Confirmed sightings go to the SightingLog (sighting_log.py) with the
rover's pose in `world_frame` from TF, and the drone's estimated position.

The logic lives in threat, geometry, draw and sighting_log; this file is only
the ROS side of it.
"""

import array
import math
import os
import time
from dataclasses import fields

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from indomitus_interfaces.msg import DroneThreat

from rover_drone_detection import geometry
from rover_drone_detection.detector import Detector, resolve_model_path
from rover_drone_detection.draw import annotate
from rover_drone_detection.image_msg import bgr_fields, to_bgr
from rover_drone_detection.sighting_log import SightingLog
from rover_drone_detection.threat import Level, ThreatParams, ThreatTracker


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
        self.hfov_deg = float(p('hfov_deg', 90.0))
        self.drone_size_m = float(p('drone_size_m', 0.3))
        self.world_frame = str(p('world_frame', 'odom'))
        self.base_frame = str(p('base_frame', 'base_footprint'))
        gps_topic = str(p('gps_topic', ''))
        log_enabled = bool(p('log_enabled', True))
        log_dir = str(p('log_dir', '~/drone_sightings'))
        log_interval = float(p('log_interval', 1.0))
        self.draw_path_zone = bool(p('draw_path_zone', True))
        self.threat_params = self._declare_params(ThreatParams)

        models_dir = os.path.join(get_package_share_directory('rover_drone_detection'), 'models')
        path = resolve_model_path(model, models_dir)
        self.get_logger().info(f'loading {path}')
        self.detector = Detector(path, conf=conf, iou=iou, imgsz=imgsz, device=device, half=half)
        t0 = time.monotonic()
        self.detector.warmup()
        self.get_logger().info(
            f'model ready ({(time.monotonic() - t0) * 1e3:.0f} ms warmup), classes '
            f'{self.detector.names}')

        self.tracker = ThreatTracker(self.threat_params)
        self.log = SightingLog(log_dir, log_interval) if log_enabled else None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._frame = None
        self._info = None
        self._fix = None
        self._level = Level.CLEAR
        self._frames = 0
        self._infer_ms = 0.0

        self.det_pub = self.create_publisher(Detection2DArray, 'drones/detections', 10)
        self.threat_pub = self.create_publisher(DroneThreat, 'drones/threat', 10)
        self.image_pub = self.create_publisher(Image, 'drones/image_annotated',
                                               qos_profile_sensor_data)
        # Depth 1 best effort: only the newest frame matters.
        self.create_subscription(Image, 'image', self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, 'camera_info', self._on_info, qos_profile_sensor_data)
        if gps_topic:
            self.create_subscription(NavSatFix, gps_topic, self._on_fix, qos_profile_sensor_data)
        self.create_timer(1.0 / max(rate, 0.1), self._tick)
        self.create_timer(10.0, self._report)

        self.get_logger().info(
            f'drone_detection_node: {self.resolve_topic_name("image")} at <= {rate:g} Hz, '
            f'log {"to " + self.log.root if self.log else "off"}')

    def _declare_params(self, cls):
        """One read-only parameter per field of cls, defaulting to the field's."""
        read_only = ParameterDescriptor(read_only=True)
        return cls(**{
            f.name: type(f.default)(self.declare_parameter(f.name, f.default, read_only).value)
            for f in fields(cls)
        })

    # --- inputs -------------------------------------------------------------

    def _on_image(self, msg):
        self._frame = msg

    def _on_info(self, msg):
        self._info = msg

    def _on_fix(self, msg):
        if msg.status.status >= msg.status.STATUS_FIX:
            self._fix = msg

    # --- per frame ----------------------------------------------------------

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

        h, w = bgr.shape[:2]
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        state = self.tracker.update(stamp, dets, w, h)
        intr = self._intrinsics(w, h)

        self.det_pub.publish(self._detections_msg(msg.header, dets))
        self.threat_pub.publish(self._threat_msg(msg.header, state, h, intr))

        log_due = self.log is not None and self.log.due(stamp, state.level)
        annotated = None
        if self.image_pub.get_subscription_count() > 0 or log_due:
            annotated = annotate(bgr, dets, state, self.threat_params, self.draw_path_zone)
        if annotated is not None and self.image_pub.get_subscription_count() > 0:
            out = Image(header=msg.header, encoding='bgr8')
            out.height, out.width, out.step, data = bgr_fields(annotated)
            out.data = array.array('B', data)
            self.image_pub.publish(out)

        if state.level != self._level:
            log = self.get_logger().info if state.level == Level.CLEAR else self.get_logger().warn
            log(f'drone threat {self._level.name} -> {state.level.name} '
                f'({state.count} in view, conf {state.confidence:.2f})')
            self._level = state.level

        if log_due:
            try:
                self.log.write(stamp, state.level,
                               self._record(msg.header, state, dets, w, h, intr), annotated)
            except OSError as exc:
                self.get_logger().error(f'cannot write sighting: {exc}',
                                        throttle_duration_sec=10.0)

    def _report(self):
        if self._frames:
            self.get_logger().info(
                f'{self._frames / 10.0:.1f} fps processed, inference {self._infer_ms:.0f} ms, '
                f'threat {self._level.name}')
        self._frames = 0

    # --- geometry -----------------------------------------------------------

    def _intrinsics(self, w, h):
        """K of camera_info scaled to this image (ros_res may differ from the
        calibrated size), or the hfov_deg pinhole without one."""
        info = self._info
        if info is None or info.k[0] <= 0 or not info.width or not info.height:
            return geometry.Intrinsics.from_camera_info(None, w, h, self.hfov_deg)
        sx, sy = w / info.width, h / info.height
        k = info.k
        return geometry.Intrinsics(k[0] * sx, k[4] * sy, k[2] * sx, k[5] * sy)

    def _target_geometry(self, state, h, intr):
        d = state.target
        if d is None:
            return math.nan, math.nan, math.nan
        return (geometry.bearing(d.cx, intr),
                geometry.depth_from_size(d.width, intr, self.drone_size_m),
                d.height / h)

    def _lookup(self, target, source, stamp):
        """(translation, rotation) of `source` in `target` at `stamp`, else latest, else None."""
        if not source:
            return None
        for when, timeout in ((Time.from_msg(stamp), 0.05), (Time(), 0.0)):
            try:
                tf = self.tf_buffer.lookup_transform(target, source, when,
                                                     timeout=Duration(seconds=timeout))
            except TransformException:
                continue
            t, r = tf.transform.translation, tf.transform.rotation
            return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)
        return None

    # --- messages -----------------------------------------------------------

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

    def _threat_msg(self, header, state, h, intr):
        bearing, depth, size = self._target_geometry(state, h, intr)
        return DroneThreat(header=header, level=int(state.level), count=state.count,
                           confidence=float(state.confidence), bearing=float(bearing),
                           size=float(size), range_est=float(depth))

    def _record(self, header, state, dets, w, h, intr):
        bearing, depth, size = self._target_geometry(state, h, intr)
        d = state.target
        rec = {
            'model': os.path.basename(self.detector.path),
            'frame_id': header.frame_id,
            'image_size': [w, h],
            'count': state.count,
            'confidence': state.confidence,
            'bbox': [d.x1, d.y1, d.x2, d.y2] if d else None,
            'bearing': bearing,
            'size': size,
            'range_est': depth,
            'detections': [{'bbox': [x.x1, x.y1, x.x2, x.y2], 'score': x.score} for x in dets],
            'rover': None,
            'drone': None,
            'gps': None,
        }
        base = self._lookup(self.world_frame, self.base_frame, header.stamp)
        if base is not None:
            (x, y, _), q = base
            rec['rover'] = {'frame': self.world_frame, 'x': x, 'y': y, 'yaw': geometry.yaw_of(q)}
        if d is not None and math.isfinite(depth):
            cam = self._lookup(self.world_frame, header.frame_id, header.stamp)
            if cam is not None:
                p = geometry.transform_point(*cam, geometry.point_in_camera(d.cx, d.y2, depth, intr))
                rec['drone'] = {'frame': self.world_frame, 'x': float(p[0]), 'y': float(p[1])}
        if self._fix is not None:
            f = self._fix
            rec['gps'] = {'lat': f.latitude, 'lon': f.longitude, 'alt': f.altitude}
        return rec


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
