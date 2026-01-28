#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import math

class GlassFusionNode(Node):
    def __init__(self):
        super().__init__('glass_fusion_node')

        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.rgb_topic = '/camera/camera/color/image_raw'
        self.cam_info_topic = '/camera/camera/depth/camera_info'
        self.radar_topic = '/acconeer/range_profile'

        self.radar_fov_rad = np.deg2rad(40.0) 
        self.glass_threshold_m = 0.1  # Depth > Radar + 0.1m = Glass threshold
        
        self.radar_min_intensity = 0.0 

        self.create_subscription(CameraInfo, self.cam_info_topic, self.info_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.create_subscription(Image, self.rgb_topic, self.rgb_cb, 10)
        self.create_subscription(PointCloud2, self.radar_topic, self.radar_cb, 10)


        self.pub_debug = self.create_publisher(Image, '/glass_detection/overlay', 10)
        self.pub_repaired_depth = self.create_publisher(Image, '/camera/camera/depth/repaired', 10)

        self.cv_bridge = CvBridge()
        self.cam_model = None
        self.latest_radar_dist = None
        self.latest_rgb = None
        self.radar_active = False

    def info_cb(self, msg):
        if self.cam_model is None:
            self.cam_model = msg
            self.get_logger().info("Camera Info Received.")

    def rgb_cb(self, msg):
        try:
            self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB Error: {e}")

    def radar_cb(self, msg):
        self.radar_active = True
        data = np.frombuffer(msg.data, dtype=np.uint8)
        point_step = msg.point_step
        num_points = msg.width
        
        max_intensity = 0.0
        peak_dist = -1.0

        for i in range(num_points):
            offset = i * point_step
            x = struct.unpack_from('<f', data, offset)[0]
            intensity = struct.unpack_from('<f', data, offset + 12)[0]

            if intensity > max_intensity:
                max_intensity = intensity
                peak_dist = x

        self.get_logger().info(f"Max Radar Intensity: {max_intensity:.1f} at {peak_dist:.2f}m")

        if max_intensity > self.radar_min_intensity:
            self.latest_radar_dist = peak_dist
        else:
            self.latest_radar_dist = None

    def depth_cb(self, msg):
        if self.latest_rgb is None:
            return 

        debug_img = self.latest_rgb.copy()
        h, w, _ = debug_img.shape

        status_color = (0, 0, 255) 
        status_text = "Init..."

        if not self.radar_active:
            status_text = "WAITING FOR RADAR..."
        elif self.latest_radar_dist is None:
            status_color = (0, 255, 255) # Yellow
            status_text = "Radar: Scanning (No Peak)"
        else:
            status_color = (0, 255, 0) # Green
            status_text = f"Target: {self.latest_radar_dist:.2f}m"

        cv2.putText(debug_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if self.cam_model is not None and self.latest_radar_dist is not None:
            try:
                fx = self.cam_model.k[0]
                fov_pixels = int((self.radar_fov_rad * fx)) 
                center_x, center_y = w // 2, h // 2
                x1 = max(0, center_x - fov_pixels // 2)
                x2 = min(w, center_x + fov_pixels // 2)
                y1 = max(0, center_y - fov_pixels // 2)
                y2 = min(h, center_y + fov_pixels // 2)

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # glass criteria
                cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_m = cv_depth.astype(float) * 0.001 
                depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
                
                roi_depth = depth_m[y1:y2, x1:x2]
                mask_invalid = (roi_depth == 0)
                mask_penetrated = (roi_depth > (self.latest_radar_dist + self.glass_threshold_m))
                glass_mask_roi = np.logical_or(mask_invalid, mask_penetrated)

                full_glass_mask = np.zeros((h, w), dtype=np.uint8)
                full_glass_mask[y1:y2, x1:x2] = glass_mask_roi.astype(np.uint8) * 255
                
                red_overlay = np.zeros_like(debug_img)
                red_overlay[:] = (0, 0, 255)
                
                debug_img = np.where(full_glass_mask[..., None] > 0, 
                                     cv2.addWeighted(debug_img, 0.7, red_overlay, 0.3, 0), 
                                     debug_img)

            except Exception as e:
                self.get_logger().error(f"Error: {e}")

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header.stamp = msg.header.stamp   # BEST: sync to depth
        debug_msg.header.frame_id = msg.header.frame_id  # optional but good
        self.pub_debug.publish(debug_msg)

        # self.pub_debug.publish(self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = GlassFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


