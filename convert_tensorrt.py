from ultralytics import YOLO
"""
# Load a YOLO11n PyTorch model
model = YOLO("yolo11x-seg.pt")

# Export the model to TensorRT with DLA enabled (only works with FP16 or INT8)
model.export(format="engine", device="cuda", half=True)  # dla:0 or dla:1 corresponds to the DLA cores # device="dla:0"
"""
# Load the exported TensorRT model
#trt_model = YOLO("yolo11x-seg-fp16.engine")
trt_model = YOLO("yolo11x-seg.pt")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg", device=0)