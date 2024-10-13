from ultralytics import YOLO

model = YOLO("yolov8n-seg.yaml").load('yolov8x.pt') # build from YAML and transfer weights

# Train the model
results = model.train(data="retina_seg.yaml", epochs=100, imgsz=224, resume=False, project='model')