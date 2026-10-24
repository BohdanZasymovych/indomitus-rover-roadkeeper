import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace

from rover_bringup.launch_utils import include_launch


ROVER_NAMESPACE = "rover"
CONTROLLERS = [
    "swerve_controller_test",
    "joint_state_broadcaster",
    "odometry_controller",
    "diff_bar_effort_controller",
]


def generate_launch_description() -> LaunchDescription:
    """Run the dirt road world with the forward RGB camera simulated, plus the road pipeline."""
    rover_sim_share = get_package_share_directory("rover_sim")
    rover_description_share = get_package_share_directory("rover_description")
    zed_description_share = get_package_share_directory("zed_description")

    world_file = os.path.join(rover_sim_share, "worlds", "dirt_road.sdf")
    controllers_yaml = os.path.join(rover_sim_share, "config", "controllers.yaml")
    bridge_yaml = os.path.join(rover_sim_share, "config", "bridge_gz.yaml")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": [
                # -s runs the server only, so no Gazebo window opens.
                PythonExpression(["'-v 4 -r ' if '", LaunchConfiguration("gui"), "' == 'true' else '-v 4 -s -r '"]),
                world_file,
            ],
            "on_exit_shutdown": "True",
        }.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"use_sim_time": True}],
        arguments=["--ros-args", "-p", f"config_file:={bridge_yaml}"],
        output="screen",
    )

    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        parameters=[{"port": 8765}],
        output="screen",
    )

    robot_state_publisher = include_launch(
        "rover_description",
        "robot_state_publisher.launch.py",
        {
            "xacro_file": os.path.join(rover_description_share, "urdf", "rover.xacro"),
            "xacro_args": (
                f"use_sim:=true controllers_yaml_path:={controllers_yaml} "
                f"rover_namespace:={ROVER_NAMESPACE} "
                "lidar_simulate_scan:=false "
                "stereo_camera_simulate_rgb:=true "
                "stereo_camera_simulate_depth:=false "
                "stereo_camera_simulate_imu:=false"
            ),
            "use_sim_time": "true",
        },
    )

    spawn_rover = Node(
        package="ros_gz_sim",
        executable="create",
        name="rover_spawner",
        arguments=[
            "-name", "indomitus_rover",
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.2",
            "-Y", "0.0",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "gui", default_value="false", choices=["true", "false"],
            description="Open the Gazebo window (false: headless server only)",
        ),
        DeclareLaunchArgument(
            "rviz", default_value="false", choices=["true", "false"],
            description="Open RViz2 with the rover model",
        ),
        DeclareLaunchArgument(
            "assist", default_value="",
            choices=["", "true", "false"],
            description="Start with the correction on or off; '' leaves it to road_assist.yaml",
        ),
        DeclareLaunchArgument(
            "debug_vis", default_value="true", choices=["true", "false"],
            description="Publish the road overlay on road_segmentation_node/debug_overlay",
        ),
        DeclareLaunchArgument(
            "device", default_value="",
            description="Segmentation device: auto | cuda | cpu; '' = road_segmentation.yaml",
        ),
        AppendEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=(
                f"{rover_sim_share}/models:"
                f"{os.path.dirname(rover_description_share)}:"
                f"{os.path.dirname(zed_description_share)}"
            ),
        ),
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=(
                f"{rover_sim_share}/models:"
                f"{os.path.dirname(rover_description_share)}:"
                f"{os.path.dirname(zed_description_share)}"
            ),
        ),
        gazebo_launch,
        foxglove,
        GroupAction([
            PushRosNamespace(ROVER_NAMESPACE),
            bridge,
            robot_state_publisher,
            TimerAction(
                period=5.0,
                actions=[GroupAction([
                    PushRosNamespace(ROVER_NAMESPACE),
                    spawn_rover,
                ])],
            ),
            include_launch("rover_localization", "ekf.launch.py", {
                "use_sim_time": "true",
            }),
            include_launch("rover_bringup", "control.launch.py", {
                "use_sim": "true",
                "controllers_yaml": controllers_yaml,
                "controllers": " ".join(CONTROLLERS),
                "inactive_controllers": "",
            }),
            include_launch("rover_drive", "twist_mux.launch.py", {
                "use_sim_time": "true",
            }),
            include_launch("rover_drive", "drive_manager.launch.py", {
                "use_sim_time": "true",
                "controller_name": "swerve_controller_test",
            }),
        ]),
        include_launch("rover_viz", "rviz.launch.py", {
            "use_sim": "true",
            "rover_namespace": ROVER_NAMESPACE,
        }, condition=IfCondition(LaunchConfiguration("rviz"))),
        # Reads the simulated /rover/cameras/zed2i/image_raw from the bridge above.
        include_launch("rover_bringup", "road_pipeline.launch.py", {
            "rover_namespace": ROVER_NAMESPACE,
            "use_cameras": "false",
            "use_sim_time": "true",
            "assist": LaunchConfiguration("assist"),
            "debug_vis": LaunchConfiguration("debug_vis"),
            "device": LaunchConfiguration("device"),
        }),
    ])
