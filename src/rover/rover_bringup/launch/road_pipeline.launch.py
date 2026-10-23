"""Camera -> road mask -> road deviation -> steering correction, on its own.

    rover_cameras ─► road_segmentation_node ─► road_deviation_node ─► road_assist_node
    image_raw         /rover/perception/        /rover/road/state      cmd_vel_remote_raw
                      road_mask                                        -> cmd_vel_remote

For testing the chain without rover.launch.py. rover.launch.py already starts
the cameras and road_assist_node, so run only the parts it does not: with
use_cameras:=false use_assist:=false, to go beside it; with everything on, to
go without it (the drive stack is then not running, so nothing moves).

    ros2 launch rover_bringup road_pipeline.launch.py
    ros2 launch rover_bringup road_pipeline.launch.py cameras:=arducam-front-bottom assist:=true
    ros2 launch rover_bringup road_pipeline.launch.py use_cameras:=false use_assist:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import PushRosNamespace
from rover_bringup.launch_utils import include_launch


def generate_launch_description():
    namespace_val = LaunchConfiguration('rover_namespace')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rover_namespace',
            default_value=EnvironmentVariable('ROVER_NAMESPACE', default_value='rover'),
            description='ROS namespace all the nodes are pushed under.',
        ),
        DeclareLaunchArgument(
            'use_cameras', default_value='true', choices=['true', 'false'],
            description='Start rover_cameras (false: another launch already does)',
        ),
        DeclareLaunchArgument(
            'cameras', default_value='arducam-front-bottom',
            description="rover_cameras names to start, comma-separated; the one road_segmentation.yaml's "
                        "image_topic reads must be among them ('' = all with autostart)",
        ),
        DeclareLaunchArgument(
            'use_perception', default_value='true', choices=['true', 'false'],
            description='Start road_segmentation_node and road_deviation_node',
        ),
        DeclareLaunchArgument(
            'device', default_value='',
            description="Segmentation device: auto | cuda | cpu; '' = road_segmentation.yaml",
        ),
        DeclareLaunchArgument(
            'debug_vis', default_value='false', choices=['true', 'false'],
            description='Publish the road overlay on road_segmentation_node/debug_overlay',
        ),
        DeclareLaunchArgument(
            'use_assist', default_value='true', choices=['true', 'false'],
            description='Start road_assist_node (false: another launch already does)',
        ),
        DeclareLaunchArgument(
            'assist', default_value='', choices=['', 'true', 'false'],
            description="Start with the correction on or off; '' leaves it to road_assist.yaml",
        ),

        GroupAction([
            PushRosNamespace(namespace_val),

            # Own GroupAction: include arguments leak into the enclosing scope.
            GroupAction([
                include_launch('rover_cameras', 'cameras.launch.py', {
                    'rover_namespace': '',   # already pushed by this group
                    'cameras': LaunchConfiguration('cameras'),
                }),
            ], condition=IfCondition(LaunchConfiguration('use_cameras'))),

            GroupAction([
                include_launch('rover_road_segmentation', 'road_segmentation.launch.py', {
                    'rover_namespace': '',
                    'device': LaunchConfiguration('device'),
                    'debug_vis': LaunchConfiguration('debug_vis'),
                }),
            ], condition=IfCondition(LaunchConfiguration('use_perception'))),

            GroupAction([
                include_launch('rover_road_assist', 'road_assist.launch.py', {
                    'rover_namespace': '',
                    'assist': LaunchConfiguration('assist'),
                }),
            ], condition=IfCondition(LaunchConfiguration('use_assist'))),
        ]),
    ])
