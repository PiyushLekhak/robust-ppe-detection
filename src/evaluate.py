import os
import csv
import torch
import datetime
import random
import numpy as np
from ultralytics import YOLO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from src.rcnn_dataset import PPEDataset, collate_fn

# ================= CONFIGURATION =================
# Paths
TEST_IMG_DIR = "data/test"
TEST_JSON = "data/test/_annotations.coco.json"
YOLO_WEIGHTS = "runs/baseline/yolov8m_baseline/weights/best.pt"
RCNN_WEIGHTS = "runs/baseline/faster_rcnn_baseline/best.pth"
OUTPUT_DIR = "runs/evaluation"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "baseline_test_results.csv")

# Model Settings
NUM_CLASSES = 4  # Background + 3 classes
CLASS_NAMES = ["Person", "Head", "Helmet"]  # Order must match dataset IDs 1,2,3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CANONICAL NAMES =================
# canonical names (lowercase keys used in dicts)
CANONICAL_CLASS_NAMES = ["person", "head", "helmet"]
PRINT_CLASS_NAMES = ["Person", "Head", "Helmet"]

# ================= UTILS =================


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
    torch.backends.cudnn.deterministic = False  # For speed
    torch.backends.cudnn.benchmark = True


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Create CSV if it doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Model",
                    "Split",
                    "mAP@50",
                    "mAP@50-95",
                    "Person_AP",
                    "Head_AP",
                    "Helmet_AP",
                    "Macro_AP",
                ]
            )


def log_to_csv(model_name, map50, map5095, per_class_ap):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure per_class_ap has 3 elements
    safe_ap = list(per_class_ap)
    while len(safe_ap) < 3:
        safe_ap.append(0.0)

    # Use canonical mapping function regardless of source; both helpers are equivalent for lists.
    named_ap = make_named_ap(safe_ap)
    macro_ap = sum(named_ap.values()) / len(named_ap)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                model_name,
                "TEST",
                f"{map50:.4f}",
                f"{map5095:.4f}",
                f"{named_ap['person']:.4f}",
                f"{named_ap['head']:.4f}",
                f"{named_ap['helmet']:.4f}",
                f"{macro_ap:.4f}",
            ]
        )
    print(f"   [Log] Saved metrics for {model_name} to {OUTPUT_CSV}")


# ================= YOLO EVALUATION =================
def evaluate_yolo():
    print("\n" + "=" * 60)
    print(">>> 1. Evaluating YOLOv8 Baseline")
    print("=" * 60)

    try:
        model = YOLO(YOLO_WEIGHTS)

        # Run Validation on Test Split
        results = model.val(
            data="data_yolo/data.yaml",
            split="test",
            imgsz=640,
            batch=12,
            verbose=False,
            plots=False,
        )

        # --- Metrics Extraction ---
        map50 = results.box.map50
        map5095 = results.box.map

        # results.box.maps contains the AP@50-95 for each class
        per_class_ap = results.box.maps.tolist()

        # Print for Console
        named_ap = make_named_ap(per_class_ap)
        macro_ap = sum(named_ap.values()) / len(named_ap)

        print(f"   mAP@50:    {map50:.4f}")
        print(f"   mAP@50-95: {map5095:.4f}")
        print("   Per-class AP:")
        for printable in PRINT_CLASS_NAMES:
            key = printable.lower()
            print(f"     {printable:7s}: {named_ap[key]:.4f}")
        print(f"   Macro-AP (mean per-class AP): {macro_ap:.4f}")

        return {
            "name": "YOLOv8",
            "map50": map50,
            "map5095": map5095,
            "per_class_ap": per_class_ap,
        }

    except Exception as e:
        print(f"   [ERROR] YOLO Evaluation Failed: {e}")
        return None


# ================= R-CNN EVALUATION =================
def evaluate_rcnn():
    print("\n" + "=" * 60)
    print(">>> 2. Evaluating Faster R-CNN Baseline")
    print("=" * 60)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, min_size=640, max_size=640
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if not os.path.exists(RCNN_WEIGHTS):
        print(f"   [ERROR] Weights not found at {RCNN_WEIGHTS}")
        return None

    model.load_state_dict(torch.load(RCNN_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_dataset = PPEDataset(
        TEST_IMG_DIR, TEST_JSON, transforms=lambda x: F.to_tensor(x)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=6, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    print("   Running Inference...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
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

    named_ap = make_named_ap(per_class_ap)
    macro_ap = sum(named_ap.values()) / len(named_ap)

    print(f"   mAP@50:    {map50:.4f}")
    print(f"   mAP@50-95: {map5095:.4f}")
    print("   Per-class AP:")
    for printable in PRINT_CLASS_NAMES:
        key = printable.lower()
        print(f"     {printable:7s}: {named_ap[key]:.4f}")
    print(f"   Macro-AP (mean per-class AP): {macro_ap:.4f}")

    return {
        "name": "Faster R-CNN",
        "map50": map50,
        "map5095": map5095,
        "per_class_ap": per_class_ap,
    }


# ================= MAIN =================
def main():
    set_seed(42)
    setup_logging()

    # Run Evaluations
    yolo_res = evaluate_yolo()
    rcnn_res = evaluate_rcnn()

    # Log to CSV
    if yolo_res:
        log_to_csv(
            yolo_res["name"],
            yolo_res["map50"],
            yolo_res["map5095"],
            yolo_res["per_class_ap"],
        )
    if rcnn_res:
        log_to_csv(
            rcnn_res["name"],
            rcnn_res["map50"],
            rcnn_res["map5095"],
            rcnn_res["per_class_ap"],
        )

    # --- FINAL REPORT ---
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Model':<15} | {'mAP@50':<10} | {'mAP@50-95':<10} | {'Person AP':<10} | {'Head AP':<10} | {'Helmet AP':<10} | {'Macro-AP':<10}"
    )
    print("-" * 80)

    # Rows
    for res in [yolo_res, rcnn_res]:
        if res:
            # convert list -> named map for robust printing
            ap_list = res["per_class_ap"]
            named = make_named_ap(
                ap_list
            )  # this mapping is canonical and works for both models
            macro = sum(named.values()) / len(named) if len(named) > 0 else 0.0
            print(
                f"{res['name']:<15} | {res['map50']:<10.4f} | {res['map5095']:<10.4f} | "
                f"{named['person']:<10.4f} | {named['head']:<10.4f} | {named['helmet']:<10.4f} | {macro:<10.4f}"
            )
    print("=" * 80)

    # Automated Insight Generation
    if yolo_res and rcnn_res:
        y_helmet = (
            yolo_res["per_class_ap"][2]
            if yolo_res and len(yolo_res["per_class_ap"]) > 2
            else 0.0
        )
        r_helmet = (
            rcnn_res["per_class_ap"][2]
            if rcnn_res and len(rcnn_res["per_class_ap"]) > 2
            else 0.0
        )

        print("\n>>> AUTOMATED INSIGHTS:")
        if r_helmet < 0.1 and y_helmet > 0.3:
            print(
                f"1. CRITICAL: Faster R-CNN failed to detect Helmets (AP: {r_helmet:.4f}), while YOLO succeeded (AP: {y_helmet:.4f})."
            )
            print(
                "   -> Hypothesis: R-CNN's Region Proposal Network might be struggling with small/dense objects like helmets."
            )
        elif r_helmet < 0.1 and y_helmet < 0.1:
            print(
                "1. CRITICAL: Both models failed on Helmets. The dataset annotations or image quality for helmets might be the root cause."
            )
        else:
            print("1. Both models detected Helmets reasonably well.")

    print(f"\n[Success] Full report saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
