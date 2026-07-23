#!/usr/bin/env python3
import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from ament_index_python.packages import get_package_share_directory

try:
    from acconeer.exptool import a121
except ImportError:
    print("Error: acconeer-exptool not installed. Run: pip install acconeer-exptool[app]")
    sys.exit(1)

class AcconeerIQNode(Node):
    def __init__(self):
        super().__init__('acconeer_iq_node')

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.port = self.get_parameter('serial_port').get_parameter_value().string_value

        self.declare_parameter('frame_id', 'radar_link')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.declare_parameter('config_name', 'acconeer_glass_config1.json')
        config_name = self.get_parameter('config_name').get_parameter_value().string_value

        # Publishers 
        self.pcl_pub = self.create_publisher(PointCloud2, '/acconeer/range_profile', 10)

        # Connection and Config
        self.get_logger().info(f"Connecting to XM125 on {self.port}...")
        try:
            pkg_share = get_package_share_directory('acconeer_ros2_driver')
            config_path = os.path.join(pkg_share, 'config', config_name)
            
            self.get_logger().info(f"Loading config from: {config_path}")
            
            with open(config_path, 'r') as f:
                json_config = f.read()
            
            self.session_config = a121.SessionConfig.from_json(json_config)
            
            self.client = a121.Client.open(serial_port=self.port)
            self.client.setup_session(self.session_config)
            self.client.start_session()
            self.get_logger().info("Radar session started successfully.")

        except Exception as e:
            self.get_logger().error(f"Radar Setup Failed: {e}")
            sys.exit(1)

        self.base_step_m = 2.5e-3  # 2.5mm per step
        self.timer = self.create_timer(0.05, self.update_radar)  # 20Hz

    def update_radar(self):
        try:
            result = self.client.get_next()
            if isinstance(result, list):
                result = result[0]
                
            frame = result.subframes[0]  #(sweeps_per_frame, num_points)
            
            sensor_config = self.session_config.groups[0][1]
            subsweep = sensor_config.subsweeps[0]
            start_p = subsweep.start_point
            step_l = subsweep.step_length
            num_p = subsweep.num_points

            distances = (start_p + np.arange(num_p) * step_l) * self.base_step_m # meters

            # Extract features from IQ data
            amplitude = np.abs(frame)  
            mean_amplitude = amplitude.mean(axis=0)  
            amplitude_variance = amplitude.var(axis=0)  
            
            # Phase stability
            phase = np.angle(frame)
            phase_unwrapped = np.unwrap(phase, axis=0)
            phase_std = np.std(phase_unwrapped, axis=0)

            # Publish
            self.publish_range_profile(distances, mean_amplitude, amplitude_variance, phase_std)
            
        except Exception as e:
            self.get_logger().error(f"Radar processing error: {e}")

    def publish_range_profile(self, distances, amplitude, variance, phase_std):
        num_points = len(distances)

        # Pack all bins as a contiguous (N, 6) float32 buffer:
        # columns are (x=distance, y=0, z=0, amplitude, variance, phase_std).
        cloud = np.zeros((num_points, 6), dtype=np.float32)
        cloud[:, 0] = distances
        cloud[:, 3] = amplitude
        cloud[:, 4] = variance
        cloud[:, 5] = phase_std

        header = Header(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='amplitude', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='variance', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='phase_std', offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2(
            header=header,
            height=1,
            width=num_points,
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=24,  # 6 floats * 4 bytes
            row_step=24 * num_points,
            data=cloud.tobytes(),
        )

        self.pcl_pub.publish(msg)
        self.get_logger().info(f"Published range profile: {num_points} bins", throttle_duration_sec=1.0)

    def destroy_node(self):
        if hasattr(self, 'client'):
            self.client.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = AcconeerIQNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()