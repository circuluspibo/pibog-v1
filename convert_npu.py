from ultralytics import YOLO

model = YOLO("./helmet-11s.pt") #yolo11s-cls.pt")

# 모델을 OpenVINO static shape로 export (NPU friendly)
model.export(format="openvino", dynamic=False, int8=True, imgsz=640)  
