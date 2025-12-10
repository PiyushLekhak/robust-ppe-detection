import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import os


IMAGE_PATH = "../data/train/000001_jpg.rf.f695422118aba8250f5a102143d82e26.jpg"


def check_gpu():
    print("-" * 30)
    print("Checking Hardware...")
    if torch.cuda.is_available():
        print(f"✅ SUCCESS: GPU Detected: {torch.cuda.get_device_name(0)}")
        print("   This project will fly on your RTX 3060!")
    else:
        print("❌ WARNING: GPU not found. Training will be slow on CPU.")
    print("-" * 30)


def test_corruption_and_enhancement():
    print("Testing 'Night Shift' Logic...")

    # 1. Load Image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ ERROR: Could not load image at {IMAGE_PATH}")
        print("   Check the path and try again.")
        return

    # Convert BGR (OpenCV) to RGB (Standard)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Simulate "Night Shift" (Corruption)
    # We darken the image significantly
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, -0.6), contrast_limit=(-0.2, -0.2), p=1
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        ]
    )
    corrupted = transform(image=img_rgb)["image"]

    # 3. Apply "Enhancement" (CLAHE)
    # We must convert to LAB color space to apply CLAHE only to Lightness (L) channel
    lab = cv2.cvtColor(corrupted, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # 4. Display Results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Raw Input")
    axes[1].imshow(corrupted)
    axes[1].set_title("2. Corrupted (Night/Noise)")
    axes[2].imshow(enhanced)
    axes[2].set_title("3. Enhanced (CLAHE Restoration)")

    for ax in axes:
        ax.axis("off")

    output_file = "smoke_test_result.png"
    plt.savefig(output_file)
    print(f"✅ SUCCESS: Logic verified. Comparison saved to '{output_file}'")
    plt.show()


def check_yolo():
    print("Checking YOLOv8 installation...")
    try:
        # Load the medium YOLOv8 model
        model = YOLO(
            "../models/yolov8m.pt"
        )  # Make sure yolov8m.pt is available or downloaded
        print("✅ SUCCESS: YOLOv8 library imported and model loaded.")
    except Exception as e:
        print(f"❌ ERROR: YOLO failed to load. {e}")


if __name__ == "__main__":
    check_gpu()
    check_yolo()
    test_corruption_and_enhancement()
