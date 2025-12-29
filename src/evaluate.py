import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from src.rcnn_dataset import PPEDataset, collate_fn
from torchvision.transforms import functional as F

# --- CONFIG ---
TEST_IMG_DIR = "data/test"
TEST_JSON = "data/test/_annotations.coco.json"
YOLO_WEIGHTS = "runs/baseline/yolov8m_baseline/weights/best.pt"
RCNN_WEIGHTS = "runs/baseline/faster_rcnn_baseline/best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_yolo():
    """Evaluate YOLO with per-class AP extraction"""
    print("\n>>> Evaluating YOLOv8 Baseline on TEST Set...")
    try:
        model = YOLO(YOLO_WEIGHTS)
        results = model.val(
            data="data_yolo/data.yaml",
            split="test",  # Change to "val" if no test split exists
            batch=8,
            verbose=True,  # Changed to True to see per-class metrics
        )

        # Extract overall metrics
        map50 = results.box.map50
        map5095 = results.box.map

        # Extract per-class AP (if available)
        # results.box.maps contains per-class AP@50-95
        per_class_ap = None
        if hasattr(results.box, "maps") and results.box.maps is not None:
            per_class_ap = results.box.maps.tolist()

        # Extract per-class AP@50 (if available)
        per_class_ap50 = None
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            per_class_ap50 = results.box.ap50.tolist()

        return {
            "map50": map50,
            "map5095": map5095,
            "per_class_ap": per_class_ap,
            "per_class_ap50": per_class_ap50,
        }
    except Exception as e:
        print(f"YOLO Eval Failed: {e}")
        return {
            "map50": 0.0,
            "map5095": 0.0,
            "per_class_ap": None,
            "per_class_ap50": None,
        }


def evaluate_rcnn():
    """Evaluate Faster R-CNN with per-class AP extraction"""
    print("\n>>> Evaluating Faster R-CNN Baseline on TEST Set...")

    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, min_size=640, max_size=640
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    model.load_state_dict(torch.load(RCNN_WEIGHTS))
    model.to(DEVICE)
    model.eval()

    # Load data
    dataset = PPEDataset(TEST_IMG_DIR, TEST_JSON, transforms=lambda x: F.to_tensor(x))
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Compute metrics
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    print("Running inference...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i % 50 == 0:
                print(f"  Batch {i}/{len(loader)}")

            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)

            metric.update(
                [{k: v.cpu() for k, v in t.items()} for t in outputs],
                [{k: v.cpu() for k, v in t.items()} for t in targets],
            )

    res = metric.compute()

    # Extract per-class AP
    map_per_class = res.get("map_per_class", torch.tensor([]))
    per_class_ap = [x.item() for x in map_per_class] if len(map_per_class) > 0 else []

    # Pad to 3 classes if needed
    while len(per_class_ap) < 3:
        per_class_ap.append(0.0)

    return {
        "map50": res["map_50"].item(),
        "map5095": res["map"].item(),
        "per_class_ap": per_class_ap,
        "per_class_ap50": None,  # Not easily extractable
    }


def main():
    print("=" * 70)
    print("FINAL BASELINE COMPARISON (TEST SET)")
    print("=" * 70)

    # Evaluate both models
    yolo_results = evaluate_yolo()
    rcnn_results = evaluate_rcnn()

    # Print overall comparison
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)
    print(f"{'Model':<20} | {'mAP@50':<12} | {'mAP@50-95':<12}")
    print("-" * 70)
    print(
        f"{'YOLOv8':<20} | {yolo_results['map50']:<12.4f} | {yolo_results['map5095']:<12.4f}"
    )
    print(
        f"{'Faster R-CNN':<20} | {rcnn_results['map50']:<12.4f} | {rcnn_results['map5095']:<12.4f}"
    )

    # Print per-class comparison
    print("\n" + "=" * 70)
    print("PER-CLASS AP@50-95")
    print("=" * 70)

    class_names = ["Person", "Head", "Helmet"]

    # YOLO per-class
    print(f"\n{'YOLOv8:':<20}")
    if yolo_results["per_class_ap"] is not None:
        for i, (name, ap) in enumerate(zip(class_names, yolo_results["per_class_ap"])):
            print(f"  {name:<12}: {ap:.4f}")
    else:
        print("  Per-class AP not available")

    # Faster R-CNN per-class
    print(f"\n{'Faster R-CNN:':<20}")
    if rcnn_results["per_class_ap"] is not None:
        for i, (name, ap) in enumerate(zip(class_names, rcnn_results["per_class_ap"])):
            print(f"  {name:<12}: {ap:.4f}")

    # Highlight the helmet detection issue
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if rcnn_results["per_class_ap"] and rcnn_results["per_class_ap"][2] < 0.1:
        print("CRITICAL: Faster R-CNN helmet detection severely impaired")
        print(f"   Helmet AP: {rcnn_results['per_class_ap'][2]:.4f}")

    if yolo_results["map5095"] > rcnn_results["map5095"]:
        diff = (yolo_results["map5095"] - rcnn_results["map5095"]) * 100
        print(f"YOLO outperforms Faster R-CNN by {diff:.1f}% on mAP@50-95")

    print("=" * 70)
    print("\nBaseline evaluation complete!")


if __name__ == "__main__":
    main()
