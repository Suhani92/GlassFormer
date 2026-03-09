#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import message_filters


class GlassFusionNode(Node):
    def __init__(self):
        super().__init__('glass_fusion_node')

        # Topics
        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.rgb_topic = '/camera/camera/color/image_raw'
        self.cam_info_topic = '/camera/camera/color/camera_info'
        self.radar_topic = '/acconeer/range_profile'

        # Parameters
        self.declare_parameter('radar_fov_deg', 40.0)
        self.declare_parameter('glass_threshold_m', 0.1)
        self.radar_min_intensity = 0.0

        # Camera info
        self.cam_model = None
        self.create_subscription(CameraInfo, self.cam_info_topic, self.info_cb, 10)

        # Synchronization
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.radar_sub = message_filters.Subscriber(self, PointCloud2, self.radar_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.radar_sub],
            queue_size=20,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Publishers
        self.pub_debug = self.create_publisher(Image, '/glass_detection/overlay', 10)
        self.pub_mask = self.create_publisher(Image, '/glass_detection/mask', 10)

        self.pub_crop_rgb = self.create_publisher(Image, '/glass_detection/cropped/rgb', 10)
        self.pub_crop_overlay = self.create_publisher(Image, '/glass_detection/cropped/overlay', 10)
        self.pub_crop_mask = self.create_publisher(Image, '/glass_detection/cropped/mask', 10)

        # Distance publisher
        self.pub_glass_distance = self.create_publisher(
            Float32,
            '/glass_detection/distance',
            10
        )

        # NEW: Radar amplitude publisher
        self.pub_radar_amplitude = self.create_publisher(
            Float32,
            '/glass_detection/radar_amplitude',
            10
        )

        self.cv_bridge = CvBridge()

    def info_cb(self, msg):
        if self.cam_model is None:
            self.cam_model = msg
            self.get_logger().info(
                f"Camera Info Received (Frame: {msg.header.frame_id})"
            )

    def sync_callback(self, rgb_msg, depth_msg, radar_msg):

        # 1. Get parameters
        radar_fov_deg = self.get_parameter('radar_fov_deg').value
        glass_thresh = self.get_parameter('glass_threshold_m').value
        radar_fov_rad = np.deg2rad(radar_fov_deg)

        # 2. Process Radar
        data = np.frombuffer(radar_msg.data, dtype=np.uint8)
        point_step = radar_msg.point_step
        num_points = radar_msg.width

        max_intensity = 0.0
        peak_dist = -1.0

        for i in range(num_points):
            offset = i * point_step
            x = struct.unpack_from('<f', data, offset)[0]
            intensity = struct.unpack_from('<f', data, offset + 12)[0]

            if intensity > max_intensity:
                max_intensity = intensity
                peak_dist = x

        # Publish distance
        dist_msg = Float32()
        if max_intensity > self.radar_min_intensity and peak_dist > 0:
            dist_msg.data = float(peak_dist)
            self.get_logger().info(
                f"Radar Dist: {peak_dist:.3f}m | Peak Amp: {max_intensity:.1f}"
            )
        else:
            dist_msg.data = -1.0
            self.get_logger().info(
                "Radar: Scanning...", throttle_duration_sec=1.0
            )

        self.pub_glass_distance.publish(dist_msg)

        # NEW: Publish radar amplitude
        amp_msg = Float32()
        if max_intensity > self.radar_min_intensity:
            amp_msg.data = float(max_intensity)
        else:
            amp_msg.data = 0.0

        self.pub_radar_amplitude.publish(amp_msg)

        # 3. Process Images
        try:
            cv_rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            debug_img = cv_rgb.copy()
            h, w, _ = debug_img.shape
        except Exception as e:
            self.get_logger().error(f"Image Conversion Error: {e}")
            return

        full_glass_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = 0, 0, w, h

        # 4. Fusion Logic
        if self.cam_model is not None and max_intensity > self.radar_min_intensity:

            try:
                fx = self.cam_model.k[0]
                cx = self.cam_model.k[2]
                cy = self.cam_model.k[5]

                center_x = int(cx)
                center_y = int(cy)

                fov_pixels = int(radar_fov_rad * fx)

                x1 = max(0, center_x - fov_pixels // 2)
                x2 = min(w, center_x + fov_pixels // 2)
                y1 = max(0, center_y - fov_pixels // 2)
                y2 = min(h, center_y + fov_pixels // 2)

                depth_m = cv_depth.astype(float) * 0.001

                if depth_m.shape[:2] != (h, w):
                    depth_m = cv2.resize(
                        depth_m, (w, h), interpolation=cv2.INTER_NEAREST
                    )

                roi_depth = depth_m[y1:y2, x1:x2]

                mask_penetrated = roi_depth > (peak_dist + glass_thresh)
                mask_invalid = roi_depth == 0

                glass_mask_roi = np.logical_or(mask_invalid, mask_penetrated)
                glass_mask_roi = glass_mask_roi.astype(np.uint8) * 255

                kernel = np.ones((5, 5), np.uint8)
                glass_mask_roi = cv2.morphologyEx(
                    glass_mask_roi, cv2.MORPH_OPEN, kernel
                )
                glass_mask_roi = cv2.morphologyEx(
                    glass_mask_roi, cv2.MORPH_CLOSE, kernel
                )

                full_glass_mask[y1:y2, x1:x2] = glass_mask_roi

                red_overlay = np.zeros_like(debug_img)
                red_overlay[:] = (0, 0, 255)

                debug_img = np.where(
                    full_glass_mask[..., None] > 0,
                    cv2.addWeighted(debug_img, 0.7, red_overlay, 0.3, 0),
                    debug_img
                )

                cv2.putText(
                    debug_img,
                    f"Dist: {peak_dist:.2f}m | Amp: {max_intensity:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            except Exception as e:
                self.get_logger().error(f"Fusion Error: {e}")

        # Publish full outputs
        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header = rgb_msg.header
        self.pub_debug.publish(debug_msg)

        mask_msg = self.cv_bridge.cv2_to_imgmsg(full_glass_mask, "mono8")
        mask_msg.header = rgb_msg.header
        self.pub_mask.publish(mask_msg)

        # Publish cropped outputs
        if x2 > x1 and y2 > y1:
            crop_rgb_msg = self.cv_bridge.cv2_to_imgmsg(
                cv_rgb[y1:y2, x1:x2], "bgr8"
            )
            crop_rgb_msg.header = rgb_msg.header
            self.pub_crop_rgb.publish(crop_rgb_msg)

            crop_overlay_msg = self.cv_bridge.cv2_to_imgmsg(
                debug_img[y1:y2, x1:x2], "bgr8"
            )
            crop_overlay_msg.header = rgb_msg.header
            self.pub_crop_overlay.publish(crop_overlay_msg)

            crop_mask_msg = self.cv_bridge.cv2_to_imgmsg(
                full_glass_mask[y1:y2, x1:x2], "mono8"
            )
            crop_mask_msg.header = rgb_msg.header
            self.pub_crop_mask.publish(crop_mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlassFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
