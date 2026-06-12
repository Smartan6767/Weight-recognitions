"""
YOLOv8 Balanced Training Script v6
(Minor Generalization Improvements)

Changes from previous version:
  ✔ Reduced unrealistic augmentations
  ✔ Better for gym scenes / faces
  ✔ Slightly improved small-object robustness
  ✔ Keeps plates slightly emphasized
  ✔ Conservative tuning (no aggressive changes)
"""

import os
import re
import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════
#  1. PATHS & BASIC CONFIG
# ══════════════════════════════════════════════
DATA_YAML   = "/home/smartan/weight_recognition/Yolo/latest_data/test.v6i.yolov8/data.yaml"

BASE_MODEL  = "yolov8n.pt"

PROJECT_DIR = "runs/train"
RUN_NAME    = "balanced_v6_minor_improvements_(1)"

# ══════════════════════════════════════════════
#  2. TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════
EPOCHS    = 120
IMGSZ     = 640
BATCH     = 16

LR0       = 0.008
LRF       = 0.005

WARMUP_EP = 5
PATIENCE  = 40

# ══════════════════════════════════════════════
#  3. CLASS COUNTS
# ══════════════════════════════════════════════
CLASS_COUNTS = {
    "bb": 552,
    "db": 527,
    "kb": 470,
    "mb": 467,
    "plates": 327,
}

# ══════════════════════════════════════════════
#  4. SLIGHT PLATES BOOST
# ══════════════════════════════════════════════
CLASS_ORDER = ["bb", "db", "kb", "mb", "plates"]

# Only slight emphasis
WEIGHT_LIST = [1.0, 1.0, 1.0, 1.0, 1.3]

# ══════════════════════════════════════════════
#  5. MLFLOW CONFIG
# ══════════════════════════════════════════════
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT = "db_kb_detection"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def sanitize_key(name: str) -> str:
    clean = re.sub(r'[^\w\s\-\.:/]', '_', name)
    return clean.replace(" ", "_")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():

    print("\n════════════════════════════════════")
    print("      YOLOv8n TRAINING STARTED")
    print("════════════════════════════════════\n")

    print("[INFO] Weight List:")
    print(WEIGHT_LIST, "\n")

    with open(DATA_YAML) as f:
        data_info = yaml.safe_load(f)

    class_names = [
        c for c in data_info.get("names")
        if c in CLASS_COUNTS
    ]

    print("[INFO] Classes:", class_names)

    with mlflow.start_run(run_name=RUN_NAME):

        # ─────────────────────────────────────
        # MLFLOW LOGGING
        # ─────────────────────────────────────
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "lr0": LR0,
            "lrf": LRF,
            "warmup_epochs": WARMUP_EP,
            "patience": PATIENCE,
            "weight_strategy": "plates_only_boost",
            "plates_weight": 1.3
        })

        # ─────────────────────────────────────
        # LOAD MODEL
        # ─────────────────────────────────────
        model = YOLO(BASE_MODEL)

        # ─────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────
        results = model.train(

            # Dataset
            data=DATA_YAML,

            # Training
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,

            # Optimization
            lr0=LR0,
            lrf=LRF,
            warmup_epochs=WARMUP_EP,

            # Early stop
            patience=PATIENCE,

            # Output
            project=PROJECT_DIR,
            name=RUN_NAME,

            save=True,
            verbose=True,

            # ═══════════════════════════════
            # AUGMENTATIONS
            # ═══════════════════════════════

            # Horizontal flip only
            fliplr=0.5,

            # ❌ Removed vertical flip
            # Helps reduce weird face/body confusion
            flipud=0.0,

            # Mild rotations only
            degrees=10,

            # Reduced shear
            shear=5,

            # Color augmentations
            hsv_h=0.015,
            hsv_s=0.15,
            hsv_v=0.15,

            # Spatial augmentations
            translate=0.08,
            scale=0.2,

            # Mild mosaic
            mosaic=0.5,

            # Disabled mixup
            mixup=0.0,

            # Slight classification emphasis
            cls=1.05,

            # Better LR scheduling
            cos_lr=True,

            # Cache for speed/stability
            cache=True
        )

        print("\n════════════════════════════════════")
        print("         TRAINING COMPLETE")
        print("════════════════════════════════════\n")

        print("Weights used:", WEIGHT_LIST)

        print("\n[INFO] Best model saved at:")
        print(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt\n")

if __name__ == "__main__":
    main()