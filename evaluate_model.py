from ultralytics import YOLO

model = YOLO("runs/detect/road_object_detection/weights/best.pt")

#Evaluate using the test dataset
results = model.val(
    data="Dataset/archive (2)/ROD-Dataset/dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8
)

print("****model evaluation completed****")
