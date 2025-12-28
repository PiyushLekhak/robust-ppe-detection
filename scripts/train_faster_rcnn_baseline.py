import os
import time
import torch
import random
import numpy as np
import torchvision
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
NUM_WORKERS = 2
PIN_MEMORY = True


# ---------------- SEEDING ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # For speed
    torch.backends.cudnn.benchmark = True  # For speed


# ---------------- TRANSFORMS ----------------
def transform(img):
    return F.to_tensor(img)


# ---------------- MODEL ----------------
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=640, max_size=640
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Freeze backbone only (matches YOLO freeze=10 philosophy)
    for param in model.backbone.body.parameters():
        param.requires_grad = False

    return model


# ---------------- TRAINING ----------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
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

    return total_loss / len(data_loader)


# ---------------- VALIDATION ----------------
@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True  # Enable per-class metrics
    )

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

        metric.update(outputs_cpu, targets_cpu)

    result = metric.compute()
    map_50 = result["map_50"].item()
    map_50_95 = result["map"].item()
    map_per_class = result.get("map_per_class", None)

    return map_50, map_50_95, map_per_class


# ---------------- MAIN ----------------
def main():
    seed_everything(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Training Faster R-CNN on {DEVICE} for {EPOCHS} epochs")
    print(f"Using 2x schedule (24 epochs) - standard for ResNet-50 backbone")

    # Datasets
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

    # Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    # Multi-step scheduler: drop LR at epoch 16 (standard for 2x schedule)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16], gamma=0.1
    )

    scaler = amp.GradScaler(enabled=USE_CUDA)

    best_map = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, DEVICE, epoch, scaler
        )

        # Validate
        map_50, map_50_95, map_per_class = validate(model, val_loader, DEVICE)

        # Update LR
        lr_scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {train_loss:.4f} | "
            f"mAP@50: {map_50:.4f} | "
            f"mAP@50-95: {map_50_95:.4f}"
        )

        # Per-class AP
        if map_per_class is not None:
            class_names = ["person", "head", "helmet"]
            for i, ap in enumerate(map_per_class.tolist()):
                if i < len(class_names):
                    print(f"  {class_names[i]} AP: {ap:.4f}")

        # Save best model
        if map_50 > best_map:
            best_map = map_50
            best_path = os.path.join(RUN_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >>> Best model saved (mAP@50: {best_map:.4f})")

    # Save last model
    last_path = os.path.join(RUN_DIR, "last.pth")
    torch.save(model.state_dict(), last_path)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best mAP@50: {best_map:.4f}")
    print(f"Models saved in: {RUN_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
