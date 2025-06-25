import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import FrameInfo
from detect.yolo11 import YOLOv11
import cv2
import numpy as np

class DetectorNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.onnx_model = "./model/yolo11n.onnx"
        self.yaml_file = "./model/coco8.yaml"
        # 初始化模型
        self.yolo11 = YOLOv11(self.onnx_model,self.yaml_file)
        self.yolo11.load_config(confidence_thres=0.5, iou_thres=0.5)
        self.yolo11.load_model("cpu")
        
        # 创建订阅者
        self.sub = self.create_subscription(FrameInfo, "camera/frame_info", self.img_detect, 10) 
        # 创建发布者
        self.pub = self.create_publisher(FrameInfo, "detector/detect", 10)
        

    def img_detect(self, msg:FrameInfo):
        # 获取图片信息
        ori_img = cv2.imdecode(np.frombuffer(msg.image.data, np.uint8),cv2.IMREAD_COLOR)
        # 获取对图片结果进行推理
        results = self.yolo11.detect(input_image=ori_img, type="engine")
        img_boxes = self.yolo11.add_detected_box(ori_img, results)
        success, encoded_image = cv2.imencode('.jpg', img_boxes)
        if not success:
            self.get_logger().error("检测后的图像编码失败")
            return
        msg.image.data = encoded_image.tobytes()
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode("detect_pub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

        
        