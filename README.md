# Robust PPE Detection

## YOLOv8 vs Faster R-CNN under real-world image degradations

This repository benchmarks two object-detection architectures—**YOLOv8m** and **Faster R-CNN**—for helmet (PPE) detection under challenging visual conditions. It also evaluates whether lightweight, classical image-processing techniques can improve detector robustness on degraded inputs.

The emphasis is on reproducibility, failure-mode analysis, and practical robustness rather than maximum clean-data accuracy.

---

## Overview

* **Models:** YOLOv8m (Ultralytics), Faster R-CNN (torchvision)
* **Dataset:** Hard Hat / helmet detection dataset (Roboflow export)  
  <https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10>
* **Corruptions tested:** low illumination, Gaussian noise, motion blur, defocus blur
* **Enhancements:** classical DSP (denoising, contrast normalization, sharpening)
* **Metrics:** mAP@0.5 (mAP@50), FPS, etc

---

## Key Findings

* Baseline performance on clean data: **mAP@50 ≈ 0.656**
* DSP-based enhancement improved robustness under severe Gaussian noise by up to **+12.6% relative mAP**
* **Defocus blur** is a major failure mode and is largely unrecoverable with classical DSP
* YOLOv8 shows greater resilience to noise and low-light artifacts; Faster R-CNN degrades more gracefully under certain blur conditions

---

## Quick start

```bash
# clone
git clone <repo-url>
cd ROBUST-PPE-DETECTION

# create virtual environment & install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# inspect dataset (place dataset exports under data/)
python src/scripts/inspect_data.py --src data/

# (optional) generate corruptions / enhancements
python src/scripts/generate_corruptions.py --out data_corrupted/
python src/scripts/generate_enhancements.py --out data_enhanced/

# train examples
python src/training/train_yolov8m_baseline.py \
  --data data_yolo/data.yaml \
  --output runs/baseline/yolov8m

python src/training/train_faster_rcnn_baseline.py \
  --data data/ \
  --output runs/baseline/faster_rcnn

# evaluate robustness
python src/evaluation/evaluate_corruptions.py \
  --model runs/baseline/yolov8m \
  --corruptions data_corrupted/ \
  --out runs/eval_corruptions/

python src/evaluation/evaluate_enhancements.py \
  --model runs/baseline/faster_rcnn \
  --enhanced data_enhanced/ \
  --out runs/eval_enhancements/
```

---

## Repository structure

```text
ROBUST-PPE-DETECTION/
├── data/                 # original dataset (train/valid/test)
├── data_corrupted/       # synthetic degradations (severity 1/3/5)
├── data_enhanced/        # enhanced outputs
├── data_yolo/            # YOLO-format dataset + data.yaml
├── models/               
├── src/                  # training, evaluation, scripts
├── runs/                 # checkpoints, logs, plots, saved weights
├── notebooks/            # EDA and analysis
├── evaluation/           # CSV metric outputs
├── requirements.txt
└── README.md
```

---

## Reproducibility

* **Random seed:** `42` (splits, corruptions, evaluation)
* **Environment:** Python 3.12; PyTorch + torchvision; Ultralytics YOLOv8m; OpenCV; Albumentations
* **Hardware:** NVIDIA RTX 3060 mobile GPU

All experiments can be reproduced using the provided scripts and configuration files.

---

## Notes & limitations

* Classical image enhancement provides measurable benefit for noise and illumination issues but limited gains for blur (especially defocus)
* Results depend on dataset and corruption severity; expect variance across other PPE datasets or camera setups
* This work demonstrates practical, low-cost preprocessing strategies; deep restoration approaches may be required for substantial blur recovery

---
