import os
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from src.rcnn_dataset import PPEDataset, collate_fn


# ---------------- CONFIG ----------------
TRAIN_IMG_DIR = "../data/train"
TRAIN_JSON = "../data/train/_annotations.coco.json"

VAL_IMG_DIR = "../data/valid"
VAL_JSON = "../data/valid/_annotations.coco.json"

BATCH_SIZE = 4
EPOCHS = 30
LR = 0.005
NUM_CLASSES = 4  # background + 3 PPE classes

RUN_DIR = "../runs/baseline/faster_rcnn_baseline"


# ---------------- TRANSFORMS ----------------
def transform(img):
    # Torchvision detection models expect images in [0,1]
    return F.to_tensor(img)


# ---------------- MODEL ----------------
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------- TRAINING ----------------
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if i % 50 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Iter [{i}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f}"
            )

    return total_loss / len(data_loader)


# ---------------- MAIN ----------------
def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training Faster R-CNN on {device}")

    os.makedirs(RUN_DIR, exist_ok=True)

    # ---- DATA ----
    train_dataset = PPEDataset(TRAIN_IMG_DIR, TRAIN_JSON, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # ---- MODEL ----
    model = get_model(NUM_CLASSES)
    model.to(device)

    # ---- OPTIMIZER ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ---- TRAIN ----
    start_time = time.time()

    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        lr_scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] " f"Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(RUN_DIR, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
