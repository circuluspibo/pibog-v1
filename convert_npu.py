from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

# 모델을 OpenVINO static shape로 export (NPU friendly)
model.export(format="openvino", dynamic=False, int8=True, imgsz=640)  