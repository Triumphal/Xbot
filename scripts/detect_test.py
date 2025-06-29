import os
import torch

from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("./model/yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Export the model
# model.export(format="onnx")
model.export(format="engine",opset=13)

