"""Road perception chain: camera -> road_segmentation_node -> road_deviation_node.

    camera image_raw ─► road_segmentation_node ─► /rover/perception/road_mask (mono8 0/1)
                                                   └► road_deviation_node ─► /rover/road/state
                                                                              └► road_assist_node

    ros2 launch rover_road_segmentation road_segmentation.launch.py
    ros2 launch rover_road_segmentation road_segmentation.launch.py debug_vis:=true
    ros2 launch rover_road_segmentation road_segmentation.launch.py \
        image_topic:=/rover/cameras/arducam_mast/image_raw with_deviation:=false

road_assist_node itself is started by rover.launch.py / road_assist.launch.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    lc = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    params = [PathJoinSubstitution([FindPackageShare('rover_road_segmentation'), 'config',
                                    'road_segmentation.yaml'])]
    overrides = {'use_sim_time': lc('use_sim_time') == 'true',
                 'publish_debug_vis': lc('debug_vis') == 'true'}
    for key in ('image_topic', 'mask_topic', 'weights_path', 'device'):
        if lc(key):
            overrides[key] = lc(key)
    params.append(overrides)

    actions = [Node(
        package='rover_road_segmentation',
        executable='road_segmentation_node',
        name='road_segmentation_node',
        parameters=params,
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )]
    if lc('with_deviation') == 'true':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('rover_road_assist'), 'launch', 'road_deviation.launch.py'])),
            launch_arguments={'mask_topic': lc('mask_topic') or '/rover/perception/road_mask'}.items(),
        ))

    namespace = lc('rover_namespace')
    return [GroupAction(([PushRosNamespace(namespace)] if namespace else []) + actions)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rover_namespace',
            default_value=EnvironmentVariable('ROVER_NAMESPACE', default_value='rover'),
            description="Namespace to push (node names and ~/debug_overlay); topics are absolute."),
        DeclareLaunchArgument('image_topic', default_value='',
                              description="Camera Image topic; '' = config/road_segmentation.yaml"),
        DeclareLaunchArgument('mask_topic', default_value='',
                              description="mono8 0/1 mask output; '' = config (/rover/perception/road_mask)"),
        DeclareLaunchArgument('weights_path', default_value='',
                              description="Head checkpoint; '' = the one installed with the package"),
        DeclareLaunchArgument('device', default_value='', description="auto | cuda | cpu; '' = config"),
        DeclareLaunchArgument('debug_vis', default_value='false', choices=['true', 'false'],
                              description='Publish the green overlay on ~/debug_overlay'),
        DeclareLaunchArgument('with_deviation', default_value='true', choices=['true', 'false'],
                              description="Also start rover_road_assist's road_deviation_node"),
        DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=_nodes),
    ])
