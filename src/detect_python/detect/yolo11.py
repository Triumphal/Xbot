## yolo检测 

import onnxruntime as ort
import cv2
import yaml 
import os
import numpy as np
from typing import List, Tuple, Literal
import time

class YOLOv11:
    """
    YOLOv11检测类
    """
    def __init__(self, onnx_model, yaml_file):
        """
        初始化,加载模型和输出的类别
        """
        self.onnx_model = onnx_model # 模型
        self.classes = self.get_class(yaml_file)["names"] ## 类别，
        self.color_palette = np.random.uniform(0,255,size=(len(self.classes),3)) # 框颜色

    def get_class(self, yaml_file):
        if not os.path.exists(yaml_file):
            print(f"{yaml_file} does not exist!!")
        with open(yaml_file,"r",encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data

    def load_model(self,type:Literal["cpu","cuda","engine"]="cpu"):
        """
        使用ort加载模型
        """
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3 # 3忽略告警
        if type == "cpu":
            providers = ["CPUExecutionProvider"]
        elif type == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif type == "engine":
            providers = ["TensorrtExecutionProvider"]
        start_time = time.perf_counter()
        self.session = ort.InferenceSession(self.onnx_model, session_options, providers=providers) 
        print(f"load model cost: {time.perf_counter() - start_time:.5f} s")
        # 得到模型的输入
        self.model_inputs = self.session.get_inputs()

        self.input_width = self.model_inputs[0].shape[2]
        self.input_height = self.model_inputs[0].shape[3]
        self.input_name = self.model_inputs[0].name
    
    def load_config(self,confidence_thres,iou_thres):
        """
        加载配置: 置信度阈值, iou阈值
        """
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

    def letterbox(self, img:np.ndarray, new_shape:Tuple[int,int]=(640,640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        输出填充, resize之后的图像和在顶部和左侧填充的多少, 因为是对称所有，输出顶部和左侧
        Args:
            img: 输入的需要resize的图像 HWC
            new_shape: 目标尺寸大小, (H,W)
        Returns:
            img: resize和padding之后的图像
            pad: padding的形状(top,left)
        """
        shape = img.shape[:2] # # current shape [height, width] 
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # Scale ratio (new / old)
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio)) # computing padding [w, h]
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad: # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        pad = (top, left)
        return img, pad

    def preprocess(self,input_image:np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
        """
        Returns:
            image_data: (1,3,H,W), 输入读取之后的图像数据
            pad: (top, left)
        """
        self.img = input_image
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        img, pad = self.letterbox(img, (self.input_width, self.input_height)) # 填充图像 HWC
        image_data = np.array(img) / 255.0 # 归一化
        image_data = np.transpose(image_data, (2, 0, 1)) # HWC ->CHW
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32) # CHW ->NCHW
        return image_data, pad

    def postprocess(self, model_output: List[np.ndarray], pad: Tuple[int,int]) -> np.ndarray:
        """
        后处理
        Args:
            model_output: 模型的输出结果
            pad: 预处理时图片的padding的尺寸(top, left)
        Return:
            result: 后处理之后的结果，[[[x1, y1, x2, y2],score,class_id],...]

        """
        outputs = np.transpose(np.squeeze(model_output[0]))

        boxes, scores, class_ids = [], [], []

        # 计算边界框的缩放因子
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # 迭代输出的结果
        for output in outputs:
            class_scores = output[4:]
            max_score = np.amax(class_scores)

            if max_score >= self.confidence_thres:
                class_id = np.argmax(class_scores)
                cx, cy, w, h = output[0], output[1], output[2], output[3] # 中心点和宽、高
                x1, y1, x2, y2 = int((cx - w/2) / gain), int((cy - h/2) / gain), int((cx + w/2) / gain), int((cy + h/2) / gain) # 转换成在原图中的坐标，xyxy格式
                scores.append(max_score)
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        result = []
        for i in indices:
            result.append([boxes[i], scores[i], class_ids[i]])
        return result

    def detect(self, input_image:np.ndarray) -> np.ndarray:
        # start_time = time.perf_counter()
        # # 加载模型
        # self.load_model(type)
        # model_time = time.perf_counter()
        
        # 数据预处理
        img_data, pad = self.preprocess(input_image)
        
        # 推理
        model_output = self.session.run(None, {self.input_name: img_data})
        
        # 后处理
        postprocess_results = self.postprocess(model_output, pad)

        
        return postprocess_results        
        

    def add_detected_box(self, img, postprocess_results):
        """
        画出检测结果:
            img: 输入图片
            postprocess_results: 后处理的结果 [xyxy,score,id]
            save_path: 输出图像保存地址
        """
        for result in postprocess_results:
            x1, y1, x2, y2 = result[0]
            score = result[1]
            class_id = result[2]
            color = self.color_palette[class_id]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.classes[class_id]}:{score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
            cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
            cv2.putText(img, label,(label_x,label_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        return img


if __name__ == "__main__":
    onnx_model = "./model/yolo11n.onnx"
    yaml_file = "./model/coco8.yaml"
    image_path = "./src/detect_1/assets/bus.jpg"
    ori_img = cv2.imread(image_path)
    yolo11 = YOLOv11(onnx_model, yaml_file) # 确定模型
    yolo11.load_config(confidence_thres=0.2, iou_thres=0.5) # 加载配置
    yolo11.load_model(type="engine")# 加载模型
    results = yolo11.detect(input_image=ori_img) # 推理图片
    img_boxes = yolo11.add_detected_box(ori_img, results)
    cv2.imwrite("./result.jpg", img_boxes)
