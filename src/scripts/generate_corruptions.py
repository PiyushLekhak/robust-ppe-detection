import os
import cv2
import json
import shutil
from pathlib import Path
import albumentations as A
import random
import numpy as np

# ---------------- CONFIGURATION ----------------
SEED = 42
SOURCE_IMG_DIR = "data/test"
SOURCE_JSON = "data/test/_annotations.coco.json"
SOURCE_YOLO_LABEL_DIR = "data_yolo/test/labels"
OUTPUT_ROOT = "data_corrupted"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# Define The "Industrial Quad" of Corruptions
# 3 severity levels: 1 (Mild), 3 (Moderate), 5 (Severe)
CORRUPTIONS = {
    "darkness": {
        1: A.RandomBrightnessContrast(brightness_limit=-0.2, contrast_limit=0, p=1),
        3: A.RandomBrightnessContrast(brightness_limit=-0.5, contrast_limit=0, p=1),
        5: A.RandomBrightnessContrast(brightness_limit=-0.7, contrast_limit=0, p=1),
    },
    "motion_blur": {
        1: A.MotionBlur(blur_limit=(3, 5), p=1),
        3: A.MotionBlur(blur_limit=(9, 11), p=1),
        5: A.MotionBlur(blur_limit=(15, 21), p=1),
    },
    "noise": {
        1: A.GaussNoise(stddev=(10.0, 30.0), p=1),
        3: A.GaussNoise(stddev=(50.0, 100.0), p=1),
        5: A.GaussNoise(stddev=(200.0, 400.0), p=1),
    },
    "defocus_blur": {
        1: A.Defocus(radius=(3, 5), alias_blur=(0.1, 0.3), p=1),
        3: A.Defocus(radius=(7, 10), alias_blur=(0.1, 0.5), p=1),
        5: A.Defocus(radius=(15, 20), alias_blur=(0.1, 0.5), p=1),
    },
}


def load_coco_json(json_path):
    """Load COCO format JSON"""
    with open(json_path, "r") as f:
        return json.load(f)


def save_coco_json(data, save_path):
    """Save COCO format JSON"""
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_dataset(corruption_name, severity, transform):
    print(f"\n{'='*60}")
    print(f"Generating: {corruption_name.upper()} | Severity Level {severity}")
    print(f"{'='*60}")

    # Setup Paths
    dest_dir = os.path.join(OUTPUT_ROOT, corruption_name, str(severity))
    dest_img_dir = os.path.join(dest_dir, "images")
    dest_label_dir = os.path.join(dest_dir, "labels")

    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # 1. Copy JSON (Annotations remain valid for pixel-level transforms)
    # Note: Geometric transforms (rotation/scale) would require bbox updates
    # But blur/noise/darkness preserve spatial relationships
    dest_json = os.path.join(dest_dir, "_annotations.coco.json")
    shutil.copy(SOURCE_JSON, dest_json)
    print(f"Copied annotations to {dest_json}")

    # 2. Copy YOLO labels
    for label_file in Path(SOURCE_YOLO_LABEL_DIR).glob("*.txt"):
        shutil.copy(label_file, dest_label_dir)
    print(f"Copied YOLO labels to {dest_label_dir}")

    # 3. Process Images
    image_files = sorted(Path(SOURCE_IMG_DIR).glob("*.jpg"))

    if len(image_files) == 0:
        print(f"WARNING: No images found in {SOURCE_IMG_DIR}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   [!] Failed to read: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Corruption directly (parameters are fixed per-call automatically)
        augmented = transform(image=image)["image"]

        # Save (Convert back to BGR for OpenCV)
        save_path = os.path.join(dest_img_dir, img_path.name)
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, augmented)

    print(f"Saved {len(image_files)} corrupted images to {dest_img_dir}")


def verify_structure():
    """Verify source data exists"""
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"[ERROR] Source image directory not found: {SOURCE_IMG_DIR}")
        return False

    if not os.path.exists(SOURCE_JSON):
        print(f"[ERROR] Source JSON not found: {SOURCE_JSON}")
        return False

    # Count images
    img_files = list(Path(SOURCE_IMG_DIR).glob("*.jpg"))
    img_files.extend(list(Path(SOURCE_IMG_DIR).glob("*.png")))

    if len(img_files) == 0:
        print(f"[ERROR] No images found in {SOURCE_IMG_DIR}")
        return False

    print(f"Found {len(img_files)} images in source directory")
    return True


def main():
    seed_everything(SEED)
    print("=" * 60)
    print("CORRUPTION GENERATION PIPELINE")
    print("=" * 60)

    # Verify source data
    if not verify_structure():
        return

    # Clean previous runs
    if os.path.exists(OUTPUT_ROOT):
        print(f"\nRemoving existing corrupted data: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    # Generate all corruption combinations
    total_configs = len(CORRUPTIONS) * 3  # 4 corruptions × 3 severity levels
    current = 0

    for c_name, levels in CORRUPTIONS.items():
        for severity, transform in levels.items():
            current += 1
            print(f"\n[{current}/{total_configs}] Processing...")
            generate_dataset(c_name, severity, transform)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Structure: {OUTPUT_ROOT}/<corruption>/<severity>/images/")
    print(f"Corruptions generated:")
    for c_name in CORRUPTIONS.keys():
        print(f"  - {c_name}: levels 1, 3, 5")


if __name__ == "__main__":
    main()
