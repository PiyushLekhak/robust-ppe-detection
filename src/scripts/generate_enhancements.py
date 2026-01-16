import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import random

# ---------------- CONFIGURATION ----------------
SEED = 42
CORRUPTED_ROOT = "data_corrupted"
ENHANCED_ROOT = "data_enhanced"

# Only enhance severities that meaningfully degrade performance
TARGET_SEVERITIES = {
    "noise": [1, 3, 5],
    "darkness": [3, 5],
    "motion_blur": [3, 5],
    "defocus_blur": [3, 5],
}


# ---------------- REPRODUCIBILITY ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ---------------- ENHANCEMENT OPERATORS ----------------
def apply_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image, clip_limit=2.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(
        image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )


def apply_unsharp_mask(image, strength="low"):
    if strength == "low":
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    else:
        return image


# ---------------- CORRUPTION-SPECIFIC WRAPPERS ----------------
def enhance_darkness(img):
    img = apply_gamma(img, gamma=1.5)
    img = apply_clahe(img, clip_limit=2.0)
    img = apply_bilateral(img, d=5, sigma_color=50, sigma_space=50)
    return img


def enhance_noise(img):
    return apply_bilateral(img, d=9, sigma_color=75, sigma_space=75)


def enhance_motion_blur(img):
    # use "low" strength to avoid over-processing.
    return apply_unsharp_mask(img, strength="low")


def enhance_defocus_blur(img):
    return apply_unsharp_mask(img, strength="low")


# ---------------- RESTORATION MAP ----------------
RESTORATION_MAP = {
    "darkness": enhance_darkness,
    "noise": enhance_noise,
    "motion_blur": enhance_motion_blur,
    "defocus_blur": enhance_defocus_blur,
}


# ---------------- DATASET PROCESSING ----------------
def process_dataset(corruption, severity, enhancement_fn):
    src_dir = os.path.join(CORRUPTED_ROOT, corruption, str(severity))
    dst_dir = os.path.join(ENHANCED_ROOT, corruption, str(severity))

    src_img_dir = os.path.join(src_dir, "images")
    src_label_dir = os.path.join(src_dir, "labels")
    dst_img_dir = os.path.join(dst_dir, "images")
    dst_label_dir = os.path.join(dst_dir, "labels")

    if not os.path.exists(src_img_dir):
        return

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 1. Copy JSON annotations (for Faster R-CNN)
    shutil.copy(
        os.path.join(src_dir, "_annotations.coco.json"),
        os.path.join(dst_dir, "_annotations.coco.json"),
    )

    # 2. Copy YOLO labels
    for label_file in Path(src_label_dir).glob("*.txt"):
        shutil.copy(label_file, dst_label_dir)

    # 3. Process Images
    img_files = list(Path(src_img_dir).glob("*.jpg"))

    print(
        f"[Enhancement] Corruption: {corruption:<13} | "
        f"Severity: {severity} | "
        f"Method: {enhancement_fn.__name__}"
    )

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        enhanced_img = enhancement_fn(img)
        cv2.imwrite(os.path.join(dst_img_dir, img_path.name), enhanced_img)


# ---------------- MAIN ----------------
def main():
    seed_everything(SEED)
    print("=" * 60)
    print("ENHANCEMENT PHASE")
    print("=" * 60)

    if os.path.exists(ENHANCED_ROOT):
        print(f"\nRemoving existing enhanced data: {ENHANCED_ROOT}")
        shutil.rmtree(ENHANCED_ROOT)

    total_configs = sum(len(severities) for severities in TARGET_SEVERITIES.values())
    current = 0

    for corruption, enhancement_fn in RESTORATION_MAP.items():
        if corruption not in TARGET_SEVERITIES:
            continue

        for severity in TARGET_SEVERITIES[corruption]:
            current += 1
            print(f"\n[{current}/{total_configs}] Processing...")
            process_dataset(corruption, severity, enhancement_fn)

    print("\n" + "=" * 60)
    print("ENHANCEMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
