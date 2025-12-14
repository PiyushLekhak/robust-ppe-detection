import json
import os
from pathlib import Path

COCO_DIR = "../data"
OUT_DIR = "../data_yolo"

# Classes to keep and their new IDs
KEEP_CLASSES = ["person", "head", "helmet"]

"""The ‘Workers’ class was removed due to semantic overlap with ‘person’, 
which can introduce label ambiguity and degrade detector convergence."""


def convert_split(split):
    img_dir = Path(COCO_DIR) / split
    out_img_dir = Path(OUT_DIR) / split / "images"
    out_lbl_dir = Path(OUT_DIR) / split / "labels"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(img_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_map = {
        cid: KEEP_CLASSES.index(name)
        for cid, name in cats.items()
        if name in KEEP_CLASSES
    }

    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_map:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img in coco["images"]:
        img_path = img_dir / img["file_name"]
        if not img_path.exists():
            continue

        os.system(f"cp {img_path} {out_img_dir}")

        h, w = img["height"], img["width"]
        label_file = out_lbl_dir / (Path(img["file_name"]).stem + ".txt")

        with open(label_file, "w") as f:
            for ann in anns_by_img.get(img["id"], []):
                x, y, bw, bh = ann["bbox"]
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h
                cls = cat_map[ann["category_id"]]
                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")


for split in ["train", "valid", "test"]:
    convert_split(split)

print("COCO to YOLO conversion done.")
