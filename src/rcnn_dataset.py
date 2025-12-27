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

        # -------------------------------------------------
        # KEEP ONLY THESE CLASSES
        # -------------------------------------------------
        self.allowed_classes = ["person", "head", "helmet"]

        # Load COCO categories
        cats = self.coco.loadCats(self.coco.getCatIds())

        # name -> coco_category_id (only allowed classes)
        self.cat_name_to_id = {
            c["name"]: c["id"] for c in cats if c["name"] in self.allowed_classes
        }

        # coco_category_id -> contiguous label (1..N)
        # background = 0 (implicit)
        self.cat_id_to_label = {
            cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_name_to_id.values())
        }

        # -------------------------------------------------
        # FILTER IMAGES WITH AT LEAST ONE VALID ANNOTATION
        # -------------------------------------------------
        valid_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # keep image only if it has at least one allowed class
            keep = False
            for ann in anns:
                if ann["category_id"] in self.cat_id_to_label:
                    keep = True
                    break

            if keep:
                valid_ids.append(img_id)

        self.ids = sorted(valid_ids)

        print(f"[PPEDataset] Loaded {len(self.ids)} images")
        print(f"[PPEDataset] Classes:")
        for name, cid in self.cat_name_to_id.items():
            print(f"  {name} -> label {self.cat_id_to_label[cid]}")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # -----------------------
        # LOAD IMAGE
        # -----------------------
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # -----------------------
        # LOAD ANNOTATIONS
        # -----------------------
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            cat_id = ann["category_id"]

            # DROP unwanted classes (e.g. worker)
            if cat_id not in self.cat_id_to_label:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[cat_id])

        # Safety: if empty, resample next image
        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))
