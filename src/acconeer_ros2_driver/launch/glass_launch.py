from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    return LaunchDescription([

        # IQ Radar Driver
        Node(
            package='acconeer_ros2_driver',
            executable='acconeer_iq_node',
            name='acconeer_iq_driver',
            output='screen',
            parameters=[
                {'serial_port': '/dev/ttyUSB0'},
                {'config_file': 'acconeer_glass_config1.json'}
            ]
        ),

        # Glass Overlay Node
        Node(
            package='acconeer_ros2_driver',
            executable='glass_basic.py',
            name='glass_node_basic',
            output='screen'
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
