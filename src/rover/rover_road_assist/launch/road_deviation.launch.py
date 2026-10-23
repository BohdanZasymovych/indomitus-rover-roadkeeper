from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mask_topic_arg = DeclareLaunchArgument(
        'mask_topic',
        default_value='/rover/perception/road_mask',
        description='mono8 road mask, nonzero = road'
    )

    deviation_topic_arg = DeclareLaunchArgument(
        'deviation_topic',
        default_value='/rover/road/state',
        description='RoadDeviation output'
    )

    return LaunchDescription([
        mask_topic_arg,
        deviation_topic_arg,
        Node(
            package='rover_road_assist',
            executable='road_deviation_node',
            name='road_deviation_node',
            output='screen',
            parameters=[{
                'mask_topic': LaunchConfiguration('mask_topic'),
                'deviation_topic': LaunchConfiguration('deviation_topic'),
            }],
        ),
    ])
