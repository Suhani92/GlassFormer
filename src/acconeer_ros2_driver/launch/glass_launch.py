from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # Basic Radar driver
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='acconeer_node', 
        #     name='acconeer_driver',
        #     output='screen',
        #     parameters=[
        #         {'serial_port': '/dev/ttyUSB0'},
        #         {'config_name': 'acconeer_glass_config1.json'} # Ensure the config file exists in config/ folder
        #     ]
        # ),

        # Detailed IQ Radar driver
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

        # # Raw IQ Radar driver
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='acconeer_raw_node',
        #     name='acconeer_raw_driver',
        #     output='screen',
        #     parameters=[
        #         {'serial_port': '/dev/ttyUSB0'},
        #         {'config_name': "acconeer_glass_config1.json"}
        #     ]
        # ),

        # # --- Glass Detection Node ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_detector.py',
        #     name='glass_detector_node',
        #     output='screen',
        #     remappings=[('/glass/radar_phasecoh', '/glass/detection_phasecoh')]
        # ),

        # # --- Glass Visualization Node ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_viz.py',
        #     name='glass_viz_node',
        #     output='screen',
        # ),

        # --- 1. Basic Version (Fixed Threshold) ---
        Node(
            package='acconeer_ros2_driver',
            executable='glass_basic.py', 
            name='glass_node_basic',
            output='screen',
            remappings=[('/glass_detection/overlay', '/glass/debug_basic')]
        ),

        # # --- 2. Adaptive Version (Distance-based Threshold) ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_adaptive.py',
        #     name='glass_node_adaptive',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_adaptive')]
        # ),

        # # --- 3. Geometric/Corridor Version ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_geometric.py',
        #     name='glass_node_geometric',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_geometric')]
        # ),

        # # --- 4. Grid/Split-View Version ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_grid.py',
        #     name='glass_node_grid',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_grid')]
        # ),

        # # --- 4. IR+RF Fusion Version ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_ir_rf.py',
        #     name='glass_node_ir_rf',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_ir_rf')]
        # ),

        # --- 5. Multi-Peak Version ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_multipeak.py',
        #     name='glass_node_multipeak',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_multipeak')]
        # ),

        # # --- 6. Depth Repair Version ---
        # Node(
        #     package='acconeer_ros2_driver',
        #     executable='glass_depth.py',
        #     name='glass_depth_node',
        #     output='screen',
        #     remappings=[('/glass_detection/overlay', '/glass/debug_depth')]
        # ),

    ])