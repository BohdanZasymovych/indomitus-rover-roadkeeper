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
    parameters = [PathJoinSubstitution([
        FindPackageShare('rover_road_assist'), 'config', 'road_assist.yaml'])]
    extra = LaunchConfiguration('extra_params_file').perform(context)
    if extra:
        parameters.append(extra)
    overrides = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}
    # Only when given, so it does not undo an `assist` set in extra_params_file.
    assist = LaunchConfiguration('assist').perform(context)
    if assist:
        overrides['assist'] = assist == 'true'
    parameters.append(overrides)

    node = Node(
        package='rover_road_assist',
        executable='road_assist_node',
        name='road_assist_node',
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
            'assist', default_value='', choices=['', 'true', 'false'],
            description="Start with the correction on or off; '' leaves it to the "
                        "params files. road_assist/toggle switches it at runtime."),
        DeclareLaunchArgument(
            'extra_params_file', default_value='',
            description='Params file loaded over config/road_assist.yaml, e.g. by a simulation.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock instead of the system clock.'),
        OpaqueFunction(function=_nodes),
    ])
