from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="../OpenCV/input/", save=True, conf=0.5)