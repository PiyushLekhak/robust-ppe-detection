import os
import csv
import torch
import datetime
import random
import numpy as np
import traceback
from ultralytics import YOLO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from src.rcnn_dataset import PPEDataset, collate_fn

# ------------------ CONFIGURATION ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # Background + 3 classes
CANONICAL_CLASS_NAMES = ["person", "head", "helmet"]
PRINT_CLASS_NAMES = ["Person", "Head", "Helmet"]

YOLO_WEIGHTS = "runs/baseline/yolov8m_baseline/weights/best.pt"
RCNN_WEIGHTS = "runs/baseline/faster_rcnn_baseline/best.pth"

OUTPUT_DIR = "runs/evaluation_corruptions"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "corruption_results.csv")

CORRUPTIONS = ["darkness", "motion_blur", "noise", "defocus_blur"]
SEVERITIES = [1, 3, 5]

YOLO_YAML_DIR = "data_yolo"
RCNN_BASE_DIR = "data/test_c"


# ------------------ UTILS ------------------
def make_named_ap(per_class_ap_list):
    named = {}
    for i, name in enumerate(CANONICAL_CLASS_NAMES):
        val = per_class_ap_list[i] if i < len(per_class_ap_list) else 0.0
        named[name] = float(val)
    return named


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Model",
                    "Corruption",
                    "Severity",
                    "mAP@50",
                    "mAP@50-95",
                    "Person_AP",
                    "Head_AP",
                    "Helmet_AP",
                    "Macro_AP",
                    "FPS",
                ]
            )


def log_to_csv(model_name, corruption, severity, map50, map5095, per_class_ap):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_ap = list(per_class_ap)
    while len(safe_ap) < 3:
        safe_ap.append(0.0)
    named_ap = make_named_ap(safe_ap)
    macro_ap = sum(named_ap.values()) / len(named_ap)
    fps = 0.0  # Always 0.0 since FPS is not measured for corruptions

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                model_name,
                corruption,
                severity,
                f"{map50:.4f}",
                f"{map5095:.4f}",
                f"{named_ap['person']:.4f}",
                f"{named_ap['head']:.4f}",
                f"{named_ap['helmet']:.4f}",
                f"{macro_ap:.4f}",
                f"{fps:.2f}",
            ]
        )
    print(
        f"   [Log] Saved metrics for {model_name} | {corruption} | Severity {severity}"
    )


# ------------------ YOLO EVALUATION ------------------
def evaluate_yolo(yaml_path):
    try:
        model = YOLO(YOLO_WEIGHTS)
        results = model.val(
            data=yaml_path,
            split="test",
            imgsz=640,
            batch=6,
            verbose=False,
            plots=False,
        )
        map50 = results.box.map50
        map5095 = results.box.map
        per_class_ap = results.box.maps.tolist()

        del model
        torch.cuda.empty_cache()

        return map50, map5095, per_class_ap
    except Exception as e:
        print(f"[ERROR] YOLO eval failed: {e}")
        traceback.print_exc()

        if "model" in locals():
            del model
        torch.cuda.empty_cache()

        return 0.0, 0.0, [0.0] * 3


# ------------------ R-CNN EVALUATION ------------------
def evaluate_rcnn(img_dir, json_file):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, min_size=640, max_size=640
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    if not os.path.exists(RCNN_WEIGHTS):
        print(f"[ERROR] R-CNN weights not found at {RCNN_WEIGHTS}")
        return 0.0, 0.0, [0.0] * 3
    model.load_state_dict(torch.load(RCNN_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    test_dataset = PPEDataset(img_dir, json_file, transforms=lambda x: F.to_tensor(x))
    test_loader = DataLoader(
        test_dataset, batch_size=6, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(
                [{k: v.cpu() for k, v in t.items()} for t in outputs],
                [{k: v.cpu() for k, v in t.items()} for t in targets],
            )
    res = metric.compute()
    map50 = res["map_50"].item()
    map5095 = res["map"].item()
    map_per_class = res.get("map_per_class", torch.tensor([]))
    per_class_ap = [x.item() for x in map_per_class]
    while len(per_class_ap) < 3:
        per_class_ap.append(0.0)

    del model
    torch.cuda.empty_cache()

    return map50, map5095, per_class_ap


# ------------------ MAIN ------------------
def main():
    set_seed(42)
    setup_logging()
    print("=" * 60)
    print("EVALUATING ALL CORRUPTIONS & SEVERITIES")
    print("=" * 60)

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            print(f"\n>>> {corruption.upper()} | Severity {severity}")

            # Paths
            yolo_yaml = os.path.join(
                YOLO_YAML_DIR, f"data_{corruption}_{severity}.yaml"
            )
            rcnn_img_dir = os.path.join(
                RCNN_BASE_DIR, corruption, str(severity), "images"
            )
            rcnn_json = os.path.join(
                RCNN_BASE_DIR, corruption, str(severity), "_annotations.coco.json"
            )

            # YOLO
            y_map50, y_map5095, y_per_class_ap = evaluate_yolo(yolo_yaml)
            log_to_csv(
                "YOLOv8",
                corruption,
                severity,
                y_map50,
                y_map5095,
                y_per_class_ap,
            )

            # R-CNN
            r_map50, r_map5095, r_per_class_ap = evaluate_rcnn(rcnn_img_dir, rcnn_json)
            log_to_csv(
                "Faster R-CNN",
                corruption,
                severity,
                r_map50,
                r_map5095,
                r_per_class_ap,
            )

    print("\n[Success] Full corruption report saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
