from ultralytics import YOLO

#loading the original pretrained model
original_model = YOLO("yolov8n.pt")

#loading the trained model
trained_model = YOLO(
    "runs/detect/road_object_detection/weights/best.pt"

)

#testing the original model
print("****TESTING ORIGINAL YOLOv8 MODEL****")
original_model.predict(
    source="images/road.jpg",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="original_comparison"
)

#now testing trained model
print("****TESTING TRAINED YOLOv8 MODEL****")
trained_model.predict(
    source="images/road.jpg",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="trained_comparison"
)

print("****MODEL COMPARISON COMPLETED****")
