import yaml
import copy
import os

base_yaml = "data_yolo/data.yaml"
out_dir = "data_enhanced"

corruptions = ["darkness", "motion_blur", "noise", "defocus_blur"]

# enhancement severities
severity_map = {
    "noise": [1, 3, 5],
    "darkness": [3, 5],
    "motion_blur": [3, 5],
    "defocus_blur": [3, 5],
}

with open(base_yaml) as f:
    base = yaml.safe_load(f)

for corruption, severities in severity_map.items():
    for s in severities:
        y = copy.deepcopy(base)
        y["test"] = (
            f"/home/lenovo/robust-ppe-detection/"
            f"data_enhanced/{corruption}/{s}/images"
        )

        out_path = os.path.join(out_dir, f"data_{corruption}_{s}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(y, f)

        print(f"[YAML] Created {out_path}")
