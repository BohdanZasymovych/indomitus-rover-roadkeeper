"""
Drone detection on one rover_cameras camera:

    cameras/<camera>/image_raw -> drone_detection_node -> drones/{detections,threat,image_annotated}

Needs `pip3 install -r requirements.txt` (ultralytics) and `git lfs pull` (the models).

    ros2 launch rover_drone_detection drone_detection.launch.py
    ros2 launch rover_drone_detection drone_detection.launch.py camera:=arducam_front_bottom \
        model:=drone_yolo26n_640.pt
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    camera = LaunchConfiguration('camera').perform(context)
    node = Node(
        package='rover_drone_detection',
        executable='drone_detection_node',
        name='drone_detection_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('rover_drone_detection'), 'config', 'drone_detection.yaml']),
            {
                'model': LaunchConfiguration('model'),
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
            },
        ],
        remappings=[
            ('image', f'cameras/{camera}/image_raw'),
            ('camera_info', f'cameras/{camera}/camera_info'),
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
            'camera', default_value='arducam_mast',
            description='rover_cameras node name (cameras.yaml name with - as _).'),
        DeclareLaunchArgument(
            'model', default_value='drone_yolo11s_640.pt',
            description='A file in the package models/ or a path (.pt/.onnx/.engine).'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock instead of the system clock.'),
        OpaqueFunction(function=_nodes),
    ])
