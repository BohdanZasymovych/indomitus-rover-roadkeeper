import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("rover_roadkeeper"), "config", "roadkeeper_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("publish_debug_image", default_value="false"),
        Node(
            package="rover_roadkeeper",
            executable="road_offset_node",
            name="road_offset_perception_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"publish_debug_image": LaunchConfiguration("publish_debug_image")},
            ],
        ),
    ])
