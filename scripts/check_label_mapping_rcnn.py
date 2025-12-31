import os
from pathlib import Path
import numpy as np
import torch
from collections import Counter
from torchvision.ops import box_iou
from ultralytics import YOLO  # not required here, but keep pattern
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.rcnn_dataset import PPEDataset
import json

# CONFIG
RCNN_WEIGHTS = "runs/baseline/faster_rcnn_baseline/best.pth"
TEST_JSON = "data/test/_annotations.coco.json"
TEST_IMG_DIR = "data/test"
IOU_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None, min_size=640, max_size=640
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)  # background + 3
model.load_state_dict(torch.load(RCNN_WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Prepare dataset
test_dataset = PPEDataset(TEST_IMG_DIR, TEST_JSON, transforms=lambda x: F.to_tensor(x))
img_paths = []
# build id->file map from COCO
with open(TEST_JSON) as f:
    coco = json.load(f)
id2file = {img["id"]: img["file_name"] for img in coco["images"]}

confusion = Counter()
totals = Counter()

for idx in range(len(test_dataset)):
    img, target = test_dataset[idx]
    file_name = id2file[int(target["image_id"].item())]
    # gt boxes and labels
    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()  # labels are 1..3
    # predict
    with torch.no_grad():
        out = model([img.to(DEVICE)])
    o = out[0]
    if o["boxes"].numel() == 0:
        for g in gt_labels:
            confusion[(int(g), -1)] += 1
            totals[int(g)] += 1
        continue

    pred_boxes = o["boxes"].cpu()
    pred_labels = o["labels"].cpu().numpy()  # labels in 1..3
    iou = box_iou(torch.tensor(gt_boxes), pred_boxes).numpy()
    for gi in range(iou.shape[0]):
        best = iou[gi].argmax()
        best_iou = iou[gi, best]
        gt_cls = int(gt_labels[gi])
        if best_iou >= IOU_THRESH:
            pred_cls = int(pred_labels[best])
            confusion[(gt_cls, pred_cls)] += 1
        else:
            confusion[(gt_cls, -1)] += 1
        totals[gt_cls] += 1

# Print confusion table
classes = sorted(
    list(
        {k[0] for k in confusion.keys()} | {k[1] for k in confusion.keys() if k[1] >= 0}
    )
)
print("GT -> Pred confusion (counts and pct by GT class):")
header = ["GT\\Pred"] + [str(c) for c in classes] + ["UNMATCHED"]
print("\t".join(header))
for gt in sorted(set(k[0] for k in confusion.keys())):
    row_total = totals[gt]
    row = [str(gt)]
    for pred in classes:
        cnt = confusion.get((gt, pred), 0)
        pct = 100.0 * cnt / row_total if row_total > 0 else 0.0
        row.append(f"{cnt} ({pct:.1f}%)")
    unmatched = confusion.get((gt, -1), 0)
    row.append(f"{unmatched} ({100.0*unmatched/row_total:.1f}%)")
    print("\t".join(row))
