import json
import os

JSON_PATH = "../data/train/_annotations.coco.json"


def inspect_coco_json():
    if not os.path.exists(JSON_PATH):
        print(f"File not found at {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print(f"Total Images: {len(data['images'])}")
    print(f"Total Annotations: {len(data['annotations'])}")

    print("\nCategories (Classes):")
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    for cid, cname in categories.items():
        print(f"  ID {cid}: {cname}")


if __name__ == "__main__":
    inspect_coco_json()
