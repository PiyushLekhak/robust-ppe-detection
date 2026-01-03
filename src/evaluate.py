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

# ------------------ CANONICAL NAMES ------------------
# canonical names (lowercase keys used in dicts)
CANONICAL_CLASS_NAMES = ["person", "head", "helmet"]
PRINT_CLASS_NAMES = ["Person", "Head", "Helmet"]


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
                    "FPS",
                ]
            )


def log_to_csv(model_name, map50, map5095, per_class_ap, fps):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure per_class_ap has 3 elements
    safe_ap = list(per_class_ap)
    while len(safe_ap) < 3:
        safe_ap.append(0.0)

    # Use canonical mapping function regardless of source
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
                f"{fps:.2f}",
            ]
        )
    print(f"   [Log] Saved metrics for {model_name} to {OUTPUT_CSV}")


# ------------------ UNIFIED FPS MEASUREMENT ------------------
def measure_fps(
    model, dataloader, device, num_batches=50, warmup_batches=5, is_yolo=False
):
    """
    Unified FPS measurement using PyTorch CUDA Events
    Works for both YOLO and Faster R-CNN

    Args:
        model: Model to measure (YOLO or Faster R-CNN)
        dataloader: DataLoader with test images
        device: torch.device
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup iterations
        is_yolo: Boolean flag to handle YOLO-specific inference

    Returns:
        float: Average FPS
    """
    print("   [FPS] Warming up GPU...")

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

            if is_yolo:
                # YOLO expects numpy arrays or image paths
                images, _ = batch
                # Convert tensors to numpy arrays (C, H, W) -> (H, W, C) and scale to 0-255
                np_images = []
                for img_tensor in images:
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        "uint8"
                    )
                    np_images.append(img_np)
                _ = model(np_images, verbose=False)
            else:
                # Faster R-CNN
                images, _ = batch
                images = [img.to(device) for img in images]
                _ = model(images)

    # Measurement using CUDA Events
    print("   [FPS] Measuring inference speed...")
    batch_times = []
    total_images = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            images, _ = batch
            batch_size = len(images)

            # Prepare images based on model type
            if is_yolo:
                # Convert tensors to numpy for YOLO
                np_images = []
                for img_tensor in images:
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        "uint8"
                    )
                    np_images.append(img_np)
                prepared_images = np_images
            else:
                # Keep as tensors for Faster R-CNN
                prepared_images = [img.to(device) for img in images]

            # Create CUDA events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Record start
            start_event.record()

            # Inference
            if is_yolo:
                _ = model(prepared_images, verbose=False)
            else:
                _ = model(prepared_images)

            # Record end
            end_event.record()

            # Wait for completion
            torch.cuda.synchronize()

            # Get elapsed time
            elapsed_ms = start_event.elapsed_time(end_event)
            batch_times.append(elapsed_ms / 1000.0)
            total_images += batch_size

    # Calculate FPS
    total_time = sum(batch_times)
    fps = total_images / total_time if total_time > 0 else 0.0

    return fps


# ------------------ YOLO EVALUATION ------------------
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
            batch=6,
            verbose=False,
            plots=False,
        )

        # --- Metrics Extraction ---
        map50 = results.box.map50
        map5095 = results.box.map
        per_class_ap = results.box.maps.tolist()

        # Create simple dataset for FPS measurement
        test_dataset_fps = PPEDataset(
            TEST_IMG_DIR, TEST_JSON, transforms=lambda x: F.to_tensor(x)
        )
        test_loader_fps = DataLoader(
            test_dataset_fps,
            batch_size=6,
            shuffle=False,
            num_workers=0,  # No multiprocessing for accurate timing
            collate_fn=collate_fn,
        )

        fps = measure_fps(
            model,
            test_loader_fps,
            DEVICE,
            num_batches=50,
            warmup_batches=5,
            is_yolo=True,
        )

        # Print for Console
        named_ap = make_named_ap(per_class_ap)
        macro_ap = sum(named_ap.values()) / len(named_ap)

        print(f"   mAP@50:    {map50:.4f}")
        print(f"   mAP@50-95: {map5095:.4f}")
        print(f"   FPS:       {fps:.2f}")
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
            "fps": fps,
        }

    except Exception as e:
        print(f"   [ERROR] YOLO Evaluation Failed: {e}")

        traceback.print_exc()
        return None


# ------------------ R-CNN EVALUATION ------------------
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

    # Accuracy evaluation
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

    # Create separate loader for FPS (no multiprocessing)
    test_loader_fps = DataLoader(
        test_dataset, batch_size=6, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    fps = measure_fps(
        model, test_loader_fps, DEVICE, num_batches=50, warmup_batches=5, is_yolo=False
    )

    named_ap = make_named_ap(per_class_ap)
    macro_ap = sum(named_ap.values()) / len(named_ap)

    print(f"   mAP@50:    {map50:.4f}")
    print(f"   mAP@50-95: {map5095:.4f}")
    print(f"   FPS:       {fps:.2f}")
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
        "fps": fps,
    }


# ------------------ MAIN ------------------
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
            yolo_res["fps"],
        )
    if rcnn_res:
        log_to_csv(
            rcnn_res["name"],
            rcnn_res["map50"],
            rcnn_res["map5095"],
            rcnn_res["per_class_ap"],
            rcnn_res["fps"],
        )

    # --- FINAL REPORT ---
    print("\n" + "=" * 90)
    print("FINAL TEST SET RESULTS")
    print("=" * 90)

    # Header
    print(
        f"{'Model':<15} | {'mAP@50':<10} | {'mAP@50-95':<10} | {'Person AP':<10} | "
        f"{'Head AP':<10} | {'Helmet AP':<10} | {'Macro-AP':<10} | {'FPS':<10}"
    )
    print("-" * 90)

    # Rows
    for res in [yolo_res, rcnn_res]:
        if res:
            ap_list = res["per_class_ap"]
            named = make_named_ap(ap_list)
            macro = sum(named.values()) / len(named) if len(named) > 0 else 0.0
            print(
                f"{res['name']:<15} | {res['map50']:<10.4f} | {res['map5095']:<10.4f} | "
                f"{named['person']:<10.4f} | {named['head']:<10.4f} | {named['helmet']:<10.4f} | "
                f"{macro:<10.4f} | {res['fps']:<10.2f}"
            )
    print("=" * 90)

    print(f"\n[Success] Full report saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
