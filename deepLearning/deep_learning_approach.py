from ultralytics import YOLO

# Load a YOLOv8 model (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO("yolov8n.pt")  # Use "n" for Nano model, adjust as needed

# Train the model
model.train(
    data="./dataset.yaml",  # Path to dataset YAML
    epochs=50,            # Number of epochs
    imgsz=960,            # Image size
    batch=11,             # Batch size
    device="cpu"            # Use GPU (set to 0) or CPU (set to 'cpu')
)

metrics = model.val()
print(metrics)  # Outputs metrics like mAP, precision, recall

results = model.predict(source="../OpenCV/input/withDarts.jpeg", save=True, conf=0.5)

#save model
model.save("best.pt")

# Check the output directory
print(f"Results saved to: {results[0].save_dir}")
