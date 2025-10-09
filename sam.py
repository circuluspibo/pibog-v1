from ultralytics import FastSAM
model_name = "FastSAM-s"
model = FastSAM(model_name)
model.export(format="openvino", half=True, int8=True) 
