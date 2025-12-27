from ultralytics import YOLO

# Load pretrained YOLOv8m
model = YOLO("models/yolov8m.pt")

model.train(
    data="data_yolo/data.yaml",
    epochs=50,
    batch=12,
    imgsz=640,
    device=0,
    amp=True,
    workers=6,
    project="runs/baseline",
    name="yolov8m_baseline",
    seed=42,
    freeze=10,
    verbose=True,
    exist_ok=False,
)
