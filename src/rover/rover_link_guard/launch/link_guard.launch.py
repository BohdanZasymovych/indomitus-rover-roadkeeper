"""
Link guard: the rover drives itself while the operator's link is down.

    cmd_vel_remote_raw silent -> link_guard_node -> cmd_vel_ext

Topic names and the limits are in config/link_guard.yaml. cmd_vel_ext is
twist_mux's lowest priority, so the operator takes over just by driving again.

    ros2 launch rover_link_guard link_guard.launch.py
    ros2 launch rover_link_guard link_guard.launch.py extra_params_file:=turn_around.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    parameters = [PathJoinSubstitution([
        FindPackageShare('rover_link_guard'), 'config', 'link_guard.yaml'])]
    extra = LaunchConfiguration('extra_params_file').perform(context)
    if extra:
        parameters.append(extra)
    parameters.append({'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)})

    node = Node(
        package='rover_link_guard',
        executable='link_guard_node',
        name='link_guard_node',
        parameters=parameters,
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
            'extra_params_file', default_value='',
            description='Params file loaded over config/link_guard.yaml, e.g. by a simulation.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock instead of the system clock.'),
        OpaqueFunction(function=_nodes),
    ])
