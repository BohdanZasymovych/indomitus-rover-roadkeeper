#!/usr/bin/env python3
"""DINOv2 road segmentation: camera Image -> mono8 road mask for rover_road_assist.

    image_topic (sensor_msgs/Image, rgb8/bgr8/mono8/...)
        -> RoadSegmenter
        -> mask_topic (sensor_msgs/Image, mono8, same height x width as the
           camera frame, step = width, 0 = no road, 1 = road)
        -> rover_road_assist road_deviation_node

The header (stamp + frame_id) of the camera frame is copied onto the mask, so
road_assist_node's max_road_age check measures the age of the *photo*, not of
the inference. With publish_debug_vis the node also publishes a visible
overlay on ~/debug_overlay (bgr8), only while someone subscribes to it.
"""
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image

from rover_road_segmentation import config as C
from rover_road_segmentation.image_codec import image_to_array, mask_fields

# Reliable, depth 1: matches road_deviation_node's best-effort (sensor data)
# subscription as well as reliable tools like rqt, and never queues stale masks.
MASK_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def default_weights():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('rover_road_segmentation'))
        candidate = share / 'weights' / C.WEIGHTS_FILENAME
        if candidate.exists():
            return str(candidate)
    except Exception:  # noqa: BLE001 - not installed: fall back to the source tree
        pass
    return str(C.DEFAULT_WEIGHTS)


class RoadSegmentationNode(Node):

    def __init__(self):
        super().__init__('road_segmentation_node')
        p = self.declare_parameter
        image_topic = p('image_topic', '/rover/cameras/arducam_front_bottom/image_raw').value
        mask_topic = p('mask_topic', '/rover/perception/road_mask').value
        weights = p('weights_path', '').value or default_weights()
        device = p('device', 'auto').value
        use_fp16 = p('use_fp16', True).value
        p('publish_debug_vis', False)
        p('debug_overlay_alpha', 0.45)
        p('prob_threshold', C.ROAD_PROB_THRESHOLD)
        p('mask_hud', True)
        p('morph_kernel_ref', C.MORPH_KERNEL_REF)
        p('lower_half_start', C.LOWER_HALF_START)
        p('min_component_fraction', C.MIN_COMPONENT_FRACTION)
        p('min_frame_brightness', C.MIN_FRAME_BRIGHTNESS)

        # torch is imported here so a missing install fails with a clear log line.
        from rover_road_segmentation.segmenter import PostProcessParams, RoadSegmenter
        self._post_cls = PostProcessParams
        self.get_logger().info(f'loading {C.DINOV2_MODEL_NAME} + head from {weights}')
        self.segmenter = RoadSegmenter(weights, device=device, use_fp16=use_fp16, post=self._post_params())
        self.add_on_set_parameters_callback(self._on_params)

        self.mask_pub = self.create_publisher(Image, mask_topic, MASK_QOS)
        self.debug_pub = self.create_publisher(Image, '~/debug_overlay', MASK_QOS)
        # Best effort, depth 1: matches the camera's reliable publisher, and a
        # slow inference drops frames instead of building a backlog.
        sub_qos = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, image_topic, self._on_image, sub_qos)

        self._warned = set()
        self._frames, self._busy, self._t_report = 0, 0.0, time.monotonic()
        self.get_logger().info(
            f'road_segmentation_node: {image_topic} -> {mask_topic} '
            f'(device={self.segmenter.device}, fp16={self.segmenter.use_fp16})')

    # --- parameters -------------------------------------------------------
    def _post_params(self, overrides=None):
        v = {n: self.get_parameter(n).value for n in (
            'prob_threshold', 'mask_hud', 'morph_kernel_ref', 'lower_half_start',
            'min_component_fraction', 'min_frame_brightness')}
        v.update(overrides or {})
        return self._post_cls(
            threshold=float(v['prob_threshold']), mask_hud=bool(v['mask_hud']),
            morph_kernel_ref=int(v['morph_kernel_ref']), lower_half_start=float(v['lower_half_start']),
            min_component_fraction=float(v['min_component_fraction']),
            min_frame_brightness=float(v['min_frame_brightness']))

    def _on_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        overrides = {prm.name: prm.value for prm in params}
        try:
            self.segmenter.post = self._post_params(overrides)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().error(text)

    # --- frames -----------------------------------------------------------
    def _on_image(self, msg):
        t0 = time.perf_counter()
        try:
            frame, order = image_to_array(msg)
        except ValueError as exc:
            self._warn_once(msg.encoding, f'dropping frames: {exc}')
            return

        mask = self.segmenter.segment(frame, order)       # (H, W) uint8 0/1

        out = Image(header=msg.header, **mask_fields(mask))
        self.mask_pub.publish(out)

        if self.get_parameter('publish_debug_vis').value and self.debug_pub.get_subscription_count() > 0:
            self._publish_debug(msg, frame, order, mask)

        self._frames += 1
        self._busy += time.perf_counter() - t0
        now = time.monotonic()
        if now - self._t_report >= 10.0:
            rate = self._frames / (now - self._t_report)
            ms = 1000.0 * self._busy / max(1, self._frames)
            self.get_logger().info(f'{rate:.1f} masks/s, {ms:.0f} ms per frame, '
                                   f'{msg.width}x{msg.height}, road {100.0 * mask.mean():.0f}%')
            self._frames, self._busy, self._t_report = 0, 0.0, now

    def _publish_debug(self, msg, frame, order, mask):
        import cv2
        from rover_road_segmentation.visualization import info_panel, overlay
        if order == 'gray':
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif order == 'rgb':
            bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = frame[:, :, :3].copy()
        vis = overlay(bgr, mask, self.get_parameter('debug_overlay_alpha').value)
        info_panel(vis, [f'{msg.width}x{msg.height}  road {100.0 * mask.mean():.1f}%'])
        h, w = vis.shape[:2]
        self.debug_pub.publish(Image(header=msg.header, height=h, width=w, encoding='bgr8',
                                     is_bigendian=0, step=3 * w, data=vis.tobytes()))


def main():
    rclpy.init()
    node = RoadSegmentationNode()
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
