from ultralytics import YOLO

#loading the YOLOV8N MODEL
model = YOLO("yolov8n.pt")

#training part using the ROD dataset from kaggle
results = model.train(
    data="Dataset/archive (2)/ROD-Dataset/dataset/data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    fraction=0.1,
    name="road_object_detection"
)

print("****TRAINING COMPLETED****")
