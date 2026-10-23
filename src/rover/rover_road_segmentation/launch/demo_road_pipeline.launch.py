"""End-to-end demo launch file for road perception & road deviation.

Launches the complete multi-node pipeline:
  1. video_publisher_node (rover_road_segmentation)
     Plays an existing video (e.g. dev_0001.mp4) into /rover/cameras/arducam_front_bottom/image_raw
  2. road_segmentation_node (rover_road_segmentation)
     Takes camera frames, computes binary DINOv2 mask, publishes /rover/perception/road_mask (mono8 0/1)
  3. road_deviation_node (rover_road_assist)
     Takes /rover/perception/road_mask, calculates lateral offset and heading error -> /rover/road/state
  4. road_deviation_visualizer_node (rover_road_segmentation)
     Combines camera feed, mask scanlines, and deviation telemetry into an interactive HUD dashboard

Usage:
  ros2 launch rover_road_segmentation demo_road_pipeline.launch.py
  ros2 launch rover_road_segmentation demo_road_pipeline.launch.py video:=track-1/videos/dev_0008.mp4
  ros2 launch rover_road_segmentation demo_road_pipeline.launch.py show:=false save_output:=true
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from rover_road_segmentation import config as C


def _nodes(context):
    video = LaunchConfiguration('video').perform(context)
    fps = LaunchConfiguration('fps').perform(context)
    show = LaunchConfiguration('show').perform(context)
    save_output = LaunchConfiguration('save_output').perform(context)
    output_path = LaunchConfiguration('output_path').perform(context)
    device = LaunchConfiguration('device').perform(context)

    # 1. Video publisher node
    pub_node = Node(
        package='rover_road_segmentation',
        executable='video_publisher_node',
        name='video_publisher_node',
        parameters=[{
            'video_path': video,
            'image_topic': '/rover/cameras/arducam_front_bottom/image_raw',
            'fps': float(fps) if fps else 0.0,
            'loop': True,
        }],
        output='screen',
    )

    # 2. Road segmentation node
    seg_params = [
        PathJoinSubstitution([
            FindPackageShare('rover_road_segmentation'),
            'config',
            'road_segmentation.yaml'
        ])
    ]
    if device:
        seg_params.append({'device': device})

    seg_node = Node(
        package='rover_road_segmentation',
        executable='road_segmentation_node',
        name='road_segmentation_node',
        parameters=seg_params,
        output='screen',
    )

    # 3. Road deviation node (from rover_road_assist)
    dev_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('rover_road_assist'),
            'launch',
            'road_deviation.launch.py'
        ])),
        launch_arguments={
            'mask_topic': '/rover/perception/road_mask',
            'deviation_topic': '/rover/road/state',
        }.items(),
    )

    # 4. Road deviation visualizer node
    vis_node = Node(
        package='rover_road_segmentation',
        executable='road_deviation_visualizer_node',
        name='road_deviation_visualizer_node',
        parameters=[{
            'image_topic': '/rover/cameras/arducam_front_bottom/image_raw',
            'mask_topic': '/rover/perception/road_mask',
            'deviation_topic': '/rover/road/state',
            'show': show.lower() == 'true',
            'save_output': save_output.lower() == 'true',
            'output_path': output_path,
        }],
        output='screen',
    )

    return [pub_node, seg_node, dev_launch, vis_node]


def generate_launch_description():
    default_video = ''
    data_dir = C.find_data_dir()
    if data_dir and (data_dir / 'videos' / 'dev_0001.mp4').exists():
        default_video = str(data_dir / 'videos' / 'dev_0001.mp4')

    return LaunchDescription([
        DeclareLaunchArgument('video', default_value=default_video,
                              description='Path to input video (dev_0001.mp4 ... dev_0012.mp4)'),
        DeclareLaunchArgument('fps', default_value='',
                              description='Playback FPS (empty = original video rate)'),
        DeclareLaunchArgument('show', default_value='true', choices=['true', 'false'],
                              description='Display live GUI window using OpenCV'),
        DeclareLaunchArgument('save_output', default_value='false', choices=['true', 'false'],
                              description='Record the dashboard into an mp4 video file'),
        DeclareLaunchArgument('output_path', default_value='outputs/demo_ros_pipeline.mp4',
                              description='Output video path if save_output is true'),
        DeclareLaunchArgument('device', default_value='',
                              description='auto | cuda | cpu'),
        OpaqueFunction(function=_nodes),
    ])
