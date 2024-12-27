from ultralytics import YOLO

# Load a pre-trained YOLO model or fine-tune on dartboard data
model = YOLO("yolov8n.pt")
results = model.predict("input/IMG_7289.jpeg", conf=0.5, save=True)

# Display results
print(results)
