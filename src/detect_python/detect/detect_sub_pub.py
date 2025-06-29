import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import FrameInfo
from detect.yolo11_onnx import YOLOv11
import cv2
import numpy as np

class DetectorNode(Node):
    def __init__(self, name):
        super().__init__(name)
        # 初始化参数
        self.initial_paramater()

        
        # 初始化模型
        self.yolo11 = YOLOv11(self.model_path,self.yaml_file)
        self.yolo11.load_config(confidence_thres=0.5, iou_thres=0.5)
        
        # 创建订阅者
        self.sub = self.create_subscription(FrameInfo, "camera/frame_info", self.img_detect, 10) 
        # 创建发布者
        self.pub = self.create_publisher(FrameInfo, "detector/detect", 10)
        
    def initial_paramater(self):
        """
        初始化参数
        """
        # 声明参数
        self.declare_parameter(name="onnx_type",value="cpu")
        self.declare_parameter(name="model_path",value="./model/yolo11n.onnx")
        self.declare_parameter(name="yaml_file",value="./model/coco8.yaml")

        # 获取参数
        self.onnx_type = self.get_parameter("onnx_type").get_parameter_value().string_value
        self.onnx_type = self.get_parameter("model_path").get_parameter_value().string_value
        self.onnx_type = self.get_parameter("yaml_file").get_parameter_value().string_value

        # 打印参数
        self.get_logger().warn(f"onnx_type is [{self.onnx_type}]")
        self.get_logger().warn(f"model_path is [{self.onnx_type}]")
        self.get_logger().warn(f"yaml_file is [{self.onnx_type}]")

    def img_detect(self, msg:FrameInfo):
        # 获取图片信息
        ori_img = cv2.imdecode(np.frombuffer(msg.image.data, np.uint8),cv2.IMREAD_COLOR)
        # 获取对图片结果进行推理
        results,_ = self.yolo11.inference(input_image=ori_img, type=self.onnx_type)
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

        
        