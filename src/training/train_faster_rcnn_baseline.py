import os
import time
import torch
import random
import numpy as np
import torchvision
import csv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch import amp
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader

from src.rcnn_dataset import PPEDataset, collate_fn

# ---------------- CONFIG ----------------
SEED = 42
TRAIN_IMG_DIR = "data/train"
TRAIN_JSON = "data/train/_annotations.coco.json"
VAL_IMG_DIR = "data/valid"
VAL_JSON = "data/valid/_annotations.coco.json"

BATCH_SIZE = 6
EPOCHS = 24  # Standard "2x schedule" for Faster R-CNN with ResNet-50
LR = 0.005
NUM_CLASSES = 4  # background + 3 PPE classes
RUN_DIR = "runs/baseline/faster_rcnn_baseline"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # For speed
        torch.backends.cudnn.benchmark = True


def transform(img):
    return F.to_tensor(img)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=640, max_size=640
    )

    # Replace the box predictor to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Freeze only the ResNet body, keep FPN trainable
    for param in model.backbone.body.parameters():
        param.requires_grad = False
    return model


def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with amp.autocast(device_type=device.type, enabled=USE_CUDA):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses.item()
    return total_loss / len(data_loader)  # average loss per batch


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(
            [{k: v.cpu() for k, v in t.items()} for t in outputs],
            [{k: v.cpu() for k, v in t.items()} for t in targets],
        )

    res = metric.compute()

    map_50 = res["map_50"].item()
    map_50_95 = res["map"].item()

    map_per_class = res.get("map_per_class", torch.tensor([]))
    class_aps = [x.item() for x in map_per_class]
    while len(class_aps) < 3:
        class_aps.append(0.0)

    return map_50, map_50_95, class_aps[0], class_aps[1], class_aps[2]


def main():
    seed_everything(SEED)
    start_time = time.time()
    print(f"Training Faster R-CNN on {DEVICE} for {EPOCHS} epochs")
    os.makedirs(RUN_DIR, exist_ok=True)

    train_dataset = PPEDataset(TRAIN_IMG_DIR, TRAIN_JSON, transforms=transform)
    val_dataset = PPEDataset(VAL_IMG_DIR, VAL_JSON, transforms=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
    )

    model = get_model(NUM_CLASSES).to(DEVICE)
    # optimize only trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16], gamma=0.1
    )
    scaler = amp.GradScaler(enabled=USE_CUDA)

    best_map = 0.0
    csv_file = os.path.join(RUN_DIR, "results.csv")

    # Initialize CSV
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "map_50",
                "map_50_95",
                "person_ap",
                "head_ap",
                "helmet_ap",
                "lr",
            ]
        )

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, scaler)
        map_50, map_50_95, p_ap, h_ap, hel_ap = validate(model, val_loader, DEVICE)
        lr_scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {train_loss:.4f} | "
            f"mAP@50: {map_50:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.5f}"
        )

        # Log to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    map_50,
                    map_50_95,
                    p_ap,
                    h_ap,
                    hel_ap,
                    optimizer.param_groups[0]["lr"],
                ]
            )

        if map_50 > best_map:
            best_map = map_50
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best.pth"))
            print(f"  >>> New best model (mAP@50={map_50:.4f}) saved")

    torch.save(model.state_dict(), os.path.join(RUN_DIR, "last.pth"))
    total_time = time.time() - start_time
    print(f"Done. Total training time: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    main()
