# YOLOv8 Gym Equipment Detection

A YOLOv8n-based object detection pipeline for detecting gym equipment in videos and images. The pipeline covers everything from training to testset evaluation.

---

## Classes

| Label | Equipment |
|-------|-----------|
| `bb` | Barbell |
| `db` | Dumbbell |
| `kb` | Kettlebell |
| `mb` | Medicine Ball |
| `plates` | Weight Plates |

---

## Pipeline Overview

```
Video(s)
   │
   ▼
extract_frames.py       → Extract frames from test videos
   │
   ▼
Sepclass.py             → Sort frames into class folders using the trained model
   │
   ▼
[ Manual Data Cleaning ] → Delete duplicates / bad frames
   │
   ▼
inference.py            → Run inference on sorted testset, get per-class detection report
   │
   ▼
testset_conf.py         → Generate confusion matrix (counts + normalised)
```

---

## Project Structure

```
├── train.py                  # YOLOv8n training script
├── extract_frames.py         # Extract frames from video files
├── Sepclass.py               # Sort frames into class folders using model
├── inference.py              # Run inference on testset, print detection report
├── testset_conf.py           # Generate confusion matrix on testset
├── requirements.txt
└── runs/
    └── train/
        └── balanced_v6_minor_improvements_(1)/
            └── weights/
                ├── best.pt
                └── last.pt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (recommended).

---

## Step 1 — Training

Train the YOLOv8n model on your dataset.

Set the correct path to your `data.yaml` inside `train.py`:

```python
DATA_YAML = "/path/to/your/data.yaml"
```

Then run:

```bash
python train.py
```

MLflow tracks all parameters and metrics. Start the MLflow UI before training:

```bash
mlflow ui --port 5000
```

### Training Config

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8n |
| Epochs | 120 |
| Image Size | 640 |
| Batch Size | 16 |
| LR0 / LRF | 0.008 / 0.005 |
| LR Schedule | Cosine decay |
| Warmup Epochs | 5 |
| Patience (early stop) | 40 |
| Class Weights | `[1.0, 1.0, 1.0, 1.0, 1.3]` — plates slightly boosted |

### Augmentation Config

| Augmentation | Value |
|-------------|-------|
| Horizontal Flip | 0.5 |
| Vertical Flip | 0.0 (disabled) |
| Rotation | 10° |
| Shear | 5 |
| Mosaic | 0.5 |
| Mixup | 0.0 (disabled) |
| HSV-H / HSV-S / HSV-V | 0.015 / 0.15 / 0.15 |
| Translate | 0.08 |
| Scale | 0.2 |

---

## Step 2 — Extract Frames from Test Video

Convert test videos into individual frames.

```bash
python extract_frames.py --video /path/to/video.mp4
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `frames/` | Output directory |
| `--format` | `jpg` | Image format: jpg / png / bmp |
| `--every` | `1` | Save every N-th frame |
| `--fps` | — | Save at this frame rate (overrides `--every`) |
| `--quality` | `95` | JPEG quality (1–100) |
| `--start` | — | Start time in seconds |
| `--end` | — | End time in seconds |
| `--prefix` | `frame` | Filename prefix |

---

## Step 3 — Sort Frames by Class

Use the trained model to sort extracted frames into class folders.

```bash
python Sepclass.py \
  --videos_dir /path/to/frames \
  --output_dir /path/to/sorted_output \
  --model runs/train/balanced_v6_minor_improvements_(1)/weights/best.pt
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--videos_dir` | required | Folder with frames or videos |
| `--output_dir` | required | Root output folder |
| `--model` | required | Path to `best.pt` |
| `--frame_interval` | `5` | Process every N-th frame |
| `--conf` | `0.4` | Detection confidence threshold |
| `--img_size` | `640` | Inference image size |
| `--save_undetected` | off | Also save frames with no detection |
| `--save_annotated` | off | Draw bounding boxes on saved frames |

Output folders created automatically:

```
output_dir/
├── bb/
├── db/
├── kb/
├── mb/
├── plates/
└── undetected/
```

> After this step, manually review and delete duplicate or bad frames before running inference.

---

## Step 4 — Inference on Testset

Run inference on the sorted testset and get a per-class detection report.

Set paths inside `inference.py`:

```python
MODEL_PATH  = "runs/train/.../weights/best.pt"
TESTSET_DIR = "/path/to/sorted_testset"
OUTPUT_DIR  = "/path/to/detections_output"
```

Then run:

```bash
python inference.py
```

Generates `inference_report.txt` with per-class recall, correct detections, and a cross-class confusion summary.

---

## Step 5 — Confusion Matrix

Generate count and normalised confusion matrix plots on the testset.

Set paths inside `testset_conf.py`:

```python
MODEL_PATH  = "runs/train/.../weights/best.pt"
TESTSET_DIR = "/path/to/sorted_testset"
OUTPUT_DIR  = "/path/to/conf_matrix_output"
```

Then run:

```bash
python testset_conf.py
```

Saves `confusion_matrix.png` (side-by-side counts + normalised) to `OUTPUT_DIR/`.

---

## Results

| Metric | Value |
|--------|-------|
| mAP50 | 0.963 |
| mAP50-95 | 0.647 |
| Precision | 0.951 |
| Recall | 0.950 |
| F1 (@ conf 0.462) | 0.95 |

### Testset Final Output

| Equipment | Detections | Ground Truth | Accuracy |
|-----------|-----------|--------------|----------|
| BB | 127 | 148 | 85.8% |
| DB | 600 | 648 | 92.6% |
| KB | 460 | 488 | 94.3% |
| MB | 53 | 64 | 82.8% |
| PLATES | 95 | 129 | 73.6% |
