"""Road-assist simulation: the rover on a generated road, with fake perception.

    ros2 launch rover_road_assist_sim road_assist_sim.launch.py
    ros2 run rover_teleop keyboard_teleop_node          # in another terminal

Starts sim_gz.launch.py on the flat_track world with the rover's ground-truth
pose published, spawns the road drawn from config/road_sim.yaml, and runs
road_sim_node (ground truth -> road/state); sim_gz's road_assist_node, with
this config's settings layered on, sits between the laptop teleop and twist_mux:

    teleop -> cmd_vel_remote_raw -> road_assist_node -> cmd_vel_remote -> twist_mux
                                          ^
    Gazebo true pose -> road_sim_node -> road/state

Everything is set in config/road_sim.yaml; noise/latency/confidence below are
quick overrides for a single run.
"""
import os
import shutil
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, OpaqueFunction, RegisterEventHandler, TimerAction,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from rover_bringup.launch_utils import include_launch

from rover_road_assist_sim.road_sdf import road_sdf
from rover_road_assist_sim.track import build_track

SIM_NODE = '/**/road_sim_node'
ASSIST_NODE = '/**/road_assist_node'


def _write(directory: str, name: str, text: str) -> str:
    path = os.path.join(directory, name)
    with open(path, 'w') as f:
        f.write(text)
    return path


def _launch(context, tmp_dir: str):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    config_path = arg('config')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    sim = config[SIM_NODE]['ros__parameters']
    namespace = arg('rover_namespace')
    model = arg('model_name')
    road_width = float(sim.get('road_width', 3.0))

    # The road Gazebo draws, from the same track road_sim_node measures against.
    t = sim['track']
    branch = t.get('branch', {})
    track = build_track(t['start'], t.get('scale', 1.0), t['main'],
                        branch.get('from', -1), branch.get('segments', []))
    road_file = _write(tmp_dir, 'road.sdf', road_sdf(track, road_width))

    # road_assist_node gets the config's own section, plus the road width
    # forced to the simulated one: a mismatch would silently rescale every
    # offset the "camera" reports.
    assist = dict(config.get(ASSIST_NODE, {}).get('ros__parameters', {}))
    assist['road_width'] = road_width
    assist_file = _write(tmp_dir, 'road_assist.yaml', yaml.safe_dump(
        {ASSIST_NODE: {'ros__parameters': assist}}))

    bridge_file = _write(tmp_dir, 'ground_truth_bridge.yaml', yaml.safe_dump([{
        'ros_topic_name': sim.get('ground_truth_topic', 'ground_truth/odom'),
        'gz_topic_name': f'/model/{model}/odometry',
        'ros_type_name': 'nav_msgs/msg/Odometry',
        'gz_type_name': 'gz.msgs.Odometry',
        'direction': 'GZ_TO_ROS',
    }]))

    overrides = {'use_sim_time': True}
    for name, key in (('noise', 'noise.enabled'), ('latency', 'latency.enabled')):
        if arg(name):
            overrides[key] = arg(name) == 'true'
    if arg('confidence'):
        overrides['confidence.mode'] = arg('confidence')

    actions = [
        include_launch('rover_sim', 'sim_gz.launch.py', {
            'world_name': 'flat_track',
            'model_name': model,
            'spawn_delay': arg('spawn_delay'),
            'extra_xacro_args': 'simulate_ground_truth:=true',
            'rover_namespace': namespace,
            'road_assist_params_file': assist_file,
        }),

        # Same delay as the rover: the world has to be up to take a model.
        TimerAction(period=float(arg('spawn_delay')), actions=[Node(
            package='ros_gz_sim', executable='create', name='road_spawner',
            arguments=['-name', 'road', '-file', road_file],
            output='screen',
        )]),

        GroupAction([
            PushRosNamespace(namespace),
            Node(
                package='ros_gz_bridge', executable='parameter_bridge',
                name='ground_truth_bridge',
                parameters=[{'use_sim_time': True}],
                arguments=['--ros-args', '-p', f'config_file:={bridge_file}'],
                output='screen',
            ),
            Node(
                package='rover_road_assist_sim', executable='road_sim_node',
                name='road_sim_node',
                parameters=[config_path, overrides],
                output='screen',
            ),
        ]),
    ]

    if arg('rviz') == 'true':
        env = {}
        on_wayland = (os.environ.get('XDG_SESSION_TYPE', '') == 'wayland'
                      or os.environ.get('WAYLAND_DISPLAY'))
        if on_wayland and 'QT_QPA_PLATFORM' not in os.environ:
            env['QT_QPA_PLATFORM'] = 'xcb'   # Ogre renders through GLX only
        actions.append(Node(
            package='rviz2', executable='rviz2', name='road_sim_rviz',
            arguments=['-d', os.path.join(
                get_package_share_directory('rover_road_assist_sim'), 'rviz', 'road_sim.rviz')],
            parameters=[{'use_sim_time': True}],
            additional_env=env,
            output='screen',
        ))
    return actions


def generate_launch_description():
    share = get_package_share_directory('rover_road_assist_sim')
    tmp_dir = tempfile.mkdtemp(prefix='road_assist_sim_')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config', default_value=os.path.join(share, 'config', 'road_sim.yaml'),
            description='Simulation config: track, topics, noise, latency, confidence.'),
        DeclareLaunchArgument(
            'noise', default_value='', choices=['', 'true', 'false'],
            description="Override noise.enabled; '' keeps the config's."),
        DeclareLaunchArgument(
            'latency', default_value='', choices=['', 'true', 'false'],
            description="Override latency.enabled; '' keeps the config's."),
        DeclareLaunchArgument(
            'confidence', default_value='', choices=['', 'fixed', 'model'],
            description="Override confidence.mode; '' keeps the config's."),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('model_name', default_value='indomitus_rover'),
        DeclareLaunchArgument('spawn_delay', default_value='5.0'),
        DeclareLaunchArgument(
            'rover_namespace',
            default_value=EnvironmentVariable('ROVER_NAMESPACE', default_value='rover')),

        OpaqueFunction(function=_launch, kwargs={'tmp_dir': tmp_dir}),
        RegisterEventHandler(OnShutdown(on_shutdown=[
            OpaqueFunction(function=lambda _: shutil.rmtree(tmp_dir, ignore_errors=True))])),
    ])
