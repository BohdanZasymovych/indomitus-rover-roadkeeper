import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("rover_roadkeeper")
    default_params_file = os.path.join(pkg_dir, "config", "roadkeeper_params.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to ROS 2 parameters file for roadkeeper_node",
    )

    node = Node(
        package="rover_roadkeeper",
        executable="roadkeeper_node",
        name="roadkeeper_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_arg,
        node,
    ])
