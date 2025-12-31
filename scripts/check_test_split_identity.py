from pathlib import Path
import json

# YOLO test images
yolo_imgs = sorted([p.name for p in Path("data_yolo/test/images").glob("*.jpg")])

# COCO test images
with open("data/test/_annotations.coco.json") as f:
    coco = json.load(f)

coco_imgs = sorted([img["file_name"] for img in coco["images"]])

assert yolo_imgs == coco_imgs, "YOLO and COCO test splits differ!"
print("Test splits are identical")
