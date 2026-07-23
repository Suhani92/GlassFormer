from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_port = LaunchConfiguration('serial_port')
    config_name = LaunchConfiguration('config_name')

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('config_name', default_value='acconeer_glass_config1.json'),

        # IQ radar driver: publishes /acconeer/range_profile
        Node(
            package='acconeer_ros2_driver',
            executable='acconeer_iq_node',
            name='acconeer_iq_driver',
            output='screen',
            parameters=[
                {'serial_port': serial_port},
                {'config_name': config_name},
            ],
        ),

        # Radar-guided mask generation (fuses radar range with RGB-D depth)
        Node(
            package='acconeer_ros2_driver',
            executable='radar_mask_node',
            name='radar_mask_node',
            output='screen',
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
