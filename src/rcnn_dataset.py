import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class PPEDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.transforms = transforms

        # All image IDs
        self.ids = list(sorted(self.coco.imgs.keys()))

        # ---- CATEGORY ID → CONTIGUOUS LABEL MAPPING ----
        # Background = 0
        # Classes     = 1..N
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

        # ---- FILTER IMAGES WITH NO ANNOTATIONS ----
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_ids.append(img_id)

        self.ids = valid_ids

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # ---- LOAD IMAGE ----
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # ---- LOAD ANNOTATIONS ----
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    # Required for variable number of boxes per image
    return tuple(zip(*batch))
