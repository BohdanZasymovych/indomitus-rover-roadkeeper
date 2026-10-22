#!/usr/bin/env python3
import time
from typing import Optional
from pathlib import Path
import numpy as np
import cv2
import torch

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Float32
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

from rover_roadkeeper.config import (
    DEFAULT_CHECKPOINT,
    KP_STEER,
    KD_STEER,
    EMA_ALPHA,
    ORIG_WIDTH,
    ORIG_HEIGHT,
)
from rover_roadkeeper.models.dinov2_segmenter import DINOv2Segmenter
from rover_roadkeeper.control.health_monitor import HealthMonitor, SystemHealthState
from rover_roadkeeper.control.path_planner import RowWiseCentroidPathPlanner
from rover_roadkeeper.control.pid_controller import PDSteeringController
from rover_roadkeeper.utils.hud_visualizer import HUDVisualizer


class RoadkeeperNode(Node if ROS2_AVAILABLE else object):
    """
    ROS 2 Node for Roadkeeper:
      - Subscribes to camera image stream
      - Infers road segmentation via DINOv2
      - Slices lookahead bands and computes steering error
      - Runs PD cruise control and link-loss health monitoring
      - Publishes geometry_msgs/Twist to /rover/cmd_vel_ext
      - Publishes annotated tactical HUD to /rover/roadkeeper/hud_image
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 (rclpy, sensor_msgs, cv_bridge) is not installed in current environment.")

        super().__init__("roadkeeper_node")

        # Declare ROS 2 parameters
        self.declare_parameter("camera_topic", "/rover/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/rover/cmd_vel_ext")
        self.declare_parameter("hud_topic", "/rover/roadkeeper/hud_image")
        self.declare_parameter("checkpoint_path", str(DEFAULT_CHECKPOINT))
        self.declare_parameter("max_linear_speed", 1.2)   # m/s (~4.3 km/h safe crawl, up to 30 km/h)
        self.declare_parameter("max_angular_speed", 0.8)  # rad/s
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Read parameters
        self.camera_topic = self.get_parameter("camera_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.hud_topic = self.get_parameter("hud_topic").value
        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.device = torch.device(self.get_parameter("device").value)

        self.get_logger().info(f"Initializing RoadkeeperNode on device {self.device}...")

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Load DINOv2 model
        self.model = DINOv2Segmenter(pretrained_backbone=True).to(self.device)
        ckpt_path = Path(self.checkpoint_path)
        if ckpt_path.exists():
            self.get_logger().info(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(str(ckpt_path), map_location=self.device)
            self.model.head.load_state_dict(state_dict.get("head_state_dict", state_dict))
        else:
            self.get_logger().warn(f"Checkpoint {ckpt_path} not found. Running with initialized head.")
        self.model.eval()

        # Initialize submodules
        self.health_monitor = HealthMonitor()
        self.path_planner = RowWiseCentroidPathPlanner()
        self.controller = PDSteeringController()
        self.visualizer = HUDVisualizer()

        # Setup Publishers & Subscribers
        self.sub_image = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile=10,
        )
        self.pub_cmd_vel = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_hud = self.create_publisher(Image, self.hud_topic, 10)

        self.last_frame_time = time.time()
        self.fps = 0.0

        self.get_logger().info(f"RoadkeeperNode ready! Listening to: {self.camera_topic}")

    def image_callback(self, msg: Image):
        now = time.time()
        dt = max(now - self.last_frame_time, 1e-4)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_frame_time = now

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to decode image message: {e}")
            return

        # 1. Perception Inference
        class_mask, road_conf_map, mean_conf = self.model.predict(frame_bgr, device=self.device)

        # 2. Health & Link Monitor
        health_state, diagnostics = self.health_monitor.update(frame_bgr, class_mask, current_time_sec=now)

        # 3. Path Planning
        plan = self.path_planner.plan(class_mask, road_conf_map)

        # 4. PD Cruise Control
        cmd = self.controller.compute(
            raw_error=plan.normalized_error,
            road_confidence=plan.road_confidence,
            is_road_visible=plan.is_road_visible,
            health_state=health_state,
            current_time_sec=now,
        )

        # 5. Publish Twist command
        twist = Twist()
        # Scale speed according to target speed percentage
        linear_v = self.max_linear_speed * (cmd.target_speed_pct / 100.0)
        # In ROS coordinate frame, positive angular z turns left; steer is positive right
        angular_z = -cmd.steer * self.max_angular_speed

        twist.linear.x = float(linear_v)
        twist.angular.z = float(angular_z)
        self.pub_cmd_vel.publish(twist)

        # 6. Render and publish HUD
        if self.pub_hud.get_subscription_count() > 0:
            hud_bgr = self.visualizer.render(
                frame_bgr=frame_bgr,
                class_mask=class_mask,
                plan=plan,
                command=cmd,
                health_state=health_state,
                fps=self.fps,
            )
            try:
                hud_msg = self.bridge.cv2_to_imgmsg(hud_bgr, encoding="bgr8")
                hud_msg.header = msg.header
                self.pub_hud.publish(hud_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish HUD image: {e}")


def main(args=None):
    if not ROS2_AVAILABLE:
        print("Error: ROS 2 is not available in this environment.")
        return

    rclpy.init(args=args)
    node = RoadkeeperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
