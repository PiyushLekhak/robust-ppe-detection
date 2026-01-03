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
        # STABLE CLASS MAPPING
        # -------------------------------------------------
        # canonical order used everywhere (semantic names)
        # lowercase to match COCO category names in annotation file
        CANONICAL_CLASS_NAMES = ["person", "head", "helmet"]
        self.allowed_classes = CANONICAL_CLASS_NAMES.copy()

        # Load COCO categories and build name -> coco_category_id map
        cats = self.coco.loadCats(self.coco.getCatIds())

        # Only keep categories that exist in the COCO file and are in the canonical list
        self.cat_name_to_id = {
            c["name"]: c["id"] for c in cats if c["name"] in self.allowed_classes
        }

        # Build coco_category_id -> contiguous label using canonical order.
        # Assign labels 1..N (0 reserved for background) and ensure the order matches CANONICAL_CLASS_NAMES.
        self.cat_id_to_label = {}
        self.label_to_cat_id = {}
        self.class_names = []
        for idx, name in enumerate(CANONICAL_CLASS_NAMES):
            cid = self.cat_name_to_id.get(name)
            if cid is not None:
                label = idx + 1  # contiguous label: 1 -> person, 2 -> head, 3 -> helmet
                self.cat_id_to_label[cid] = label
                self.label_to_cat_id[label] = cid
                self.class_names.append(name)

        # Print mapping for reproducibility in logs
        print(f"[PPEDataset] Canonical class order: {CANONICAL_CLASS_NAMES}")
        print("[PPEDataset] name -> coco_id mapping:")
        for n in CANONICAL_CLASS_NAMES:
            cid = self.cat_name_to_id.get(n)
            if cid is not None:
                print(f"  {n} -> coco_id {cid} -> label {self.cat_id_to_label[cid]}")
            else:
                print(f"  {n} -> NOT FOUND in annotations")
        # -------------------------------------------------

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
