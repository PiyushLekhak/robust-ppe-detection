import yaml
import copy

base_yaml = "data_yolo/data.yaml"
out_dir = "data_corrupted"

corruptions = ["darkness", "motion_blur", "noise", "defocus_blur"]
severities = [1, 3, 5]

with open(base_yaml) as f:
    base = yaml.safe_load(f)

for c in corruptions:
    for s in severities:
        y = copy.deepcopy(base)
        y["test"] = f"/home/lenovo/robust-ppe-detection/data_corrupted/{c}/{s}/images"
        out_path = f"{out_dir}/data_{c}_{s}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(y, f)
        print(f"Created {out_path}")
