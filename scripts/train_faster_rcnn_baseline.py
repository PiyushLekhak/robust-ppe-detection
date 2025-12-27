import os
import time
import torch
import random
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch import amp

from src.rcnn_dataset import PPEDataset, collate_fn

# ---------------- CONFIG ----------------
SEED = 42
TRAIN_IMG_DIR = "data/train"
TRAIN_JSON = "data/train/_annotations.coco.json"

BATCH_SIZE = 4
EPOCHS = 20
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
        torch.backends.cudnn.benchmark = True


# ---------------- TRANSFORMS ----------------
def transform(img):
    # Faster R-CNN expects images in [0,1]; internal normalization applied automatically
    return F.to_tensor(img)


# ---------------- MODEL ----------------
def get_model(num_classes):
    # Load model with standard resolution (matches YOLO)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=640, max_size=640
    )

    # 1. Setup the Head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Freeze ONLY the Backbone (ResNet Body)
    # This leaves the FPN (Neck) and Heads trainable
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

        with amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

        if i % 50 == 0:
            print(
                f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f}"
            )

    return total_loss / len(data_loader)


# ---------------- MAIN ----------------
def main():
    seed_everything(SEED)
    print(f"Training Faster R-CNN (Frozen Backbone) on {DEVICE}")
    os.makedirs(RUN_DIR, exist_ok=True)

    train_dataset = PPEDataset(TRAIN_IMG_DIR, TRAIN_JSON, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
    )

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = amp.GradScaler(enabled=USE_CUDA)

    start_time = time.time()

    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(
            model, optimizer, train_loader, DEVICE, epoch, scaler
        )
        lr_scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] " f"Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(RUN_DIR, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
