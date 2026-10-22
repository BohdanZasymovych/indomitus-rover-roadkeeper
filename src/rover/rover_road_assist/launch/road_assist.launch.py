"""
Road assist: the filter between the operator's command and twist_mux.

    cmd_vel_remote_raw + road/state -> road_assist_node -> cmd_vel_remote

Both topic names are set in config/road_assist.yaml. The laptop teleop
publishes cmd_vel_remote_raw, so without this node it cannot drive: both
rover.launch.py and sim_gz.launch.py start it.

    ros2 launch rover_road_assist road_assist.launch.py assist:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    node = Node(
        package='rover_road_assist',
        executable='road_assist_node',
        name='road_assist_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('rover_road_assist'), 'config', 'road_assist.yaml']),
            {
                'assist': ParameterValue(LaunchConfiguration('assist'), value_type=bool),
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
            },
        ],
        output='screen',
    )

    namespace = LaunchConfiguration('rover_namespace').perform(context)
    if not namespace:
        return [node]
    return [GroupAction([PushRosNamespace(namespace), node])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rover_namespace',
            default_value=EnvironmentVariable('ROVER_NAMESPACE', default_value='rover'),
            description="Namespace to push; '' when the including launch already pushed it.",
        ),
        DeclareLaunchArgument(
            'assist', default_value='false', choices=['true', 'false'],
            description='Start with the correction on; road_assist/toggle switches it.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock instead of the system clock.'),
        OpaqueFunction(function=_nodes),
    ])
