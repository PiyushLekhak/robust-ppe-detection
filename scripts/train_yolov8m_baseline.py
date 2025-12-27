from ultralytics import YOLO

model = YOLO("../models/yolov8m.pt")

model.train(
    data="../data_yolo/data.yaml",
    epochs=50,
    batch=10,
    imgsz=640,
    device=0,
    amp=True,
    workers=4,
    project="../runs/baseline",
    name="yolov8m_baseline",
    seed=42,
    verbose=True,
    exist_ok=False,
)
