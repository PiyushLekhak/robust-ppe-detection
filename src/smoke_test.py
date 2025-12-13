# improved_smoke_test.py
import cv2, os, json, random, time
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMAGE_PATH = "../data/train/000001_jpg.rf.f695422118aba8250f5a102143d82e26.jpg"
MODEL_LOCAL = "../models/yolov8m.pt"
OUT_IMG = "smoke_test_result.png"
SEED = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def check_gpu():
    print("-" * 30)
    print("Checking Hardware...")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found. Using CPU.")
    print("-" * 30)


def add_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_corruption(img):
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, -0.6), contrast_limit=(-0.2, -0.2), p=1
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ]
    )
    corrupted = transform(image=img)["image"]
    corrupted = add_gaussian_noise(corrupted, sigma=15)
    return corrupted


def apply_enhancement(img):
    # conditional gamma + CLAHE
    if img.mean() < 110:
        gamma = 1.6
        img = np.clip(((img / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced


def draw_predictions_on_image(img, preds):
    canvas = img.copy()
    for det in preds:
        x1, y1, x2, y2, score, cls = det
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{cls}:{score:.2f}",
            (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return canvas


def test_corruption_and_enhancement():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Could not load image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    corrupted = apply_corruption(img_rgb)
    enhanced = apply_enhancement(corrupted)

    model = YOLO(MODEL_LOCAL)  # local path or auto-download
    # run inference on enhanced to test pipeline
    results = model(enhanced, imgsz=640, device=0)  # single image inference
    # parse boxes
    preds = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            preds.append((x1, y1, x2, y2, score, cls))

    canvas = draw_predictions_on_image(enhanced.copy(), preds)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(corrupted)
    axes[1].set_title("Corrupted")
    axes[1].axis("off")
    axes[2].imshow(enhanced)
    axes[2].set_title("Enhanced")
    axes[2].axis("off")
    axes[3].imshow(canvas)
    axes[3].set_title("Enhanced + YOLO preds")
    axes[3].axis("off")
    plt.savefig(OUT_IMG)
    print(f" Saved {OUT_IMG}")


if __name__ == "__main__":
    check_gpu()
    test_corruption_and_enhancement()
