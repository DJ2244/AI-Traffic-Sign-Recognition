from ultralytics import YOLO

# LOADING THE MODEL
model = YOLO("yolov8n.pt")

# DETECTING OBJECTS IN THE IMAGE
results = model.predict(
    source="images/road.jpg",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="predict"
)

#DISPLAYING THE DETECTED OBJECTS
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(f"{object_name}: {confidence:.2f}")

print("****Object detection completed now showing confidence levels****")
