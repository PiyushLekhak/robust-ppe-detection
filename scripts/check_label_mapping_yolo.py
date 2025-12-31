from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import box_iou
from collections import Counter

# CONFIG
DATA_YAML = "data_yolo/data.yaml"
TEST_IMG_DIR = "data_yolo/test/images"
TEST_LABEL_DIR = "data_yolo/test/labels"  # match your test labels
YOLO_WEIGHTS = "runs/baseline/yolov8m_baseline/weights/best.pt"
IOU_THRESH = 0.5
CONF_THRES = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_yolo_label(txt_path, img_w, img_h):
    boxes = []
    classes = []
    with open(txt_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.array([], dtype=int)
    return np.array(boxes), np.array(classes, dtype=int)


def main():
    model = YOLO(YOLO_WEIGHTS)
    model.to(DEVICE)
    img_paths = sorted(Path(TEST_IMG_DIR).glob("*.jpg"))
    confusion = Counter()  # (gt_cls, pred_cls) -> count
    totals = Counter()  # gt_cls -> total GT boxes considered

    for p in img_paths:
        img = p.as_posix()
        # load image to get width/height
        import cv2

        im = cv2.imread(img)
        if im is None:
            continue
        h, w = im.shape[:2]
        label_path = Path(TEST_LABEL_DIR) / (p.stem + ".txt")
        gt_boxes, gt_classes = read_yolo_label(label_path, w, h)
        if gt_boxes.shape[0] == 0:
            continue
        # YOLO predict single image
        res = model.predict(
            source=img, imgsz=640, conf=CONF_THRES, iou=0.5, verbose=False
        )
        # results for first (and only) image
        preds = res[0].boxes
        if len(preds) == 0:
            pred_boxes = np.zeros((0, 4))
            pred_classes = np.array([], dtype=int)
        else:
            # ultralytics: .xyxy, .cls (tensor)
            pred_boxes = preds.xyxy.cpu().numpy()
            pred_classes = preds.cls.cpu().numpy().astype(int)

        # compute IoU matrix and greedy match
        if pred_boxes.shape[0] == 0:
            # all GT unmatched -> count as predicted=-1
            for gt_cls in gt_classes:
                confusion[(gt_cls, -1)] += 1
                totals[gt_cls] += 1
            continue

        iou = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes)).numpy()
        for gi in range(iou.shape[0]):
            best = iou[gi].argmax()
            best_iou = iou[gi, best]
            gt_cls = int(gt_classes[gi])
            if best_iou >= IOU_THRESH:
                pred_cls = int(pred_classes[best])
                confusion[(gt_cls, pred_cls)] += 1
            else:
                confusion[(gt_cls, -1)] += 1
            totals[gt_cls] += 1

    # print confusion table
    classes = sorted(
        list(
            {k[0] for k in confusion.keys()}
            | {k[1] for k in confusion.keys() if k[1] >= 0}
        )
    )
    print("GT -> Pred confusion (counts and pct by GT class):")
    header = ["GT\\Pred"] + [str(c) for c in classes] + ["UNMATCHED"]
    print("\t".join(header))
    for gt in sorted(set(k[0] for k in confusion.keys())):
        row = [str(gt)]
        row_total = totals[gt]
        for pred in classes:
            cnt = confusion.get((gt, pred), 0)
            pct = 100.0 * cnt / row_total if row_total > 0 else 0.0
            row.append(f"{cnt} ({pct:.1f}%)")
        unmatched = confusion.get((gt, -1), 0)
        row.append(f"{unmatched} ({100.0*unmatched/row_total:.1f}%)")
        print("\t".join(row))


if __name__ == "__main__":
    main()
