from ultralytics import YOLO
import numpy as np
# Loop through all YOLOv8 model sizes
# for size in ("n", "s", "m", "l", "x"):
path = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\denseblueberry\train_blueberry_3520\train2_rep3\weights\best.pt'
# Load a YOLOv8 PyTorch model
# model = YOLO(f"yolov8{size}.pt")
model = YOLO(path)
# Export the PyTorch model to CoreML INT8 format with NMS layers
# model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])
model.export(format="coreml", int8=True, nms=True, imgsz=3520)