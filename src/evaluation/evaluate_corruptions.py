import subprocess
import sys

# Define your experiment grid
corruptions = ["darkness", "motion_blur", "noise", "defocus_blur"]
severities = [1, 3, 5]

# Path to python executable
python_exec = sys.executable

print("=" * 70)
print("ROBUSTNESS EVALUATION PIPELINE")
print("=" * 70)
print(f"Total configurations: {len(corruptions) * len(severities)}")
print(f"Python: {python_exec}")
print("=" * 70)

failed_runs = []

for corruption in corruptions:
    for severity in severities:
        # Construct Paths
        img_dir = f"data/test_c/{corruption}/{severity}/images"
        json_path = f"data/test_c/{corruption}/{severity}/_annotations.coco.json"

        # Tag for the CSV log (e.g., "darkness_lvl3")
        tag = f"{corruption}_lvl{severity}"

        print(f"\n{'='*70}")
        print(f"[Running] {tag.upper()}")
        print(f"{'='*70}")

        # Build Command
        cmd = [
            python_exec,
            "-m",
            "src.evaluation.evaluate",
            "--img_dir",
            img_dir,
            "--json_path",
            json_path,
            "--tag",
            tag,
        ]

        # Run and wait for completion
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"{tag} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {tag}")
            failed_runs.append(tag)
        except FileNotFoundError:
            print(f"ERROR: evaluate.py not found. Check path.")
            break

# Summary
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(
    f"Successful: {len(corruptions)*len(severities) - len(failed_runs)}/{len(corruptions)*len(severities)}"
)
if failed_runs:
    print(f"Failed runs: {', '.join(failed_runs)}")
print("=" * 70)
