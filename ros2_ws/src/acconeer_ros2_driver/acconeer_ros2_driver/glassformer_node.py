import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer

# The model definition lives in the `glassformer` Python package.
# Install it first with `pip install -e .` from the repo root.
from glassformer.models.glassformer import GlassSegFormerRGBRadar


class GlassFormer(Node):
    def __init__(self):
        super().__init__("GlassFormer")

        self.rgb_sub = Subscriber(
            self,
            Image,
            "/glass_detection/cropped/rgb"
        )

        self.radar_sub = Subscriber(
            self,
            Image,
            "/glass_detection/cropped/mask"
        )
        self.seg_pub = self.create_publisher(Image, '/GlassFormer/glass_segmentation', 10)
        self.rgb_image = None
        self.radar_mask = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Checkpoint path is a ROS parameter (no hardcoded paths).
        self.declare_parameter("model_path", "")
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            raise ValueError(
                "Set the 'model_path' parameter to your GlassFormer checkpoint (.pt)."
            )

        self.model = GlassSegFormerRGBRadar().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.bridge = CvBridge()
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.radar_sub],
            queue_size=10,
            slop=0.05
        )
        self.model.eval()
        print("[Glassformer] model setup done: ")
        self.sync.registerCallback(self.inference_callback)

    def preprocess(self, rgb, radar):

        IMG_SIZE = 402

        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        radar = cv2.resize(radar, (IMG_SIZE, IMG_SIZE))

        rgb = rgb.astype(np.float32) / 255.0
        radar = radar.astype(np.float32) / 255.0

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        radar = torch.from_numpy(radar).unsqueeze(0).unsqueeze(0)

        return rgb.to(self.device), radar.to(self.device)

    def inference_callback(self, rgb_msg, radar_msg):

        try:

            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            radar = self.bridge.imgmsg_to_cv2(radar_msg, desired_encoding="mono8")
            rgb_t, radar_t = self.preprocess(rgb, radar)
            with torch.no_grad():

                logits = self.model(rgb_t, radar_t)

                probs = torch.sigmoid(logits)

                pred = (probs > 0.5).float()

            pred = pred.squeeze().cpu().numpy()

            pred = (pred * 255).astype(np.uint8)
            pred = cv2.resize(
                pred,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            msg = self.bridge.cv2_to_imgmsg(pred, encoding="mono8")
            msg.header = rgb_msg.header

            self.seg_pub.publish(msg)

        except Exception as e:

            self.get_logger().error(str(e))

def main():

    rclpy.init()

    node = GlassFormer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()