from ultralytics import YOLO

#loading my trained model
model = YOLO("runs/detect/road_object_detection/weights/best.pt")

#Now time to test my road image
results = model.predict(
    source="images/road.jpg",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="trained_model_test"
)

print("****TRAINED MODEL TESTING COMPLETED****")