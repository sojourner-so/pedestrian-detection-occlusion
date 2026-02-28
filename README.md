# Pedestrian Detection with Occlusion Handling
### A Computer Vision Project | Python · YOLOv8 · Google Colab

---

## Overview

This project investigates the problem of **pedestrian detection under occlusion** — the challenge of reliably identifying people in scenes where they are partially hidden by other pedestrians, objects, or the edges of the frame. Using the Penn-Fudan Pedestrian Dataset, a YOLOv8 object detection model was fine-tuned, systematically evaluated across varying levels of occlusion, and iteratively improved through a custom data augmentation pipeline.

The project follows an end-to-end machine learning workflow: from raw data acquisition and exploratory analysis, through preprocessing, training, structured evaluation, and augmentation-driven retraining. Critically, this work treats honest failure analysis as a first-class outcome — identifying where and why the model struggles is treated as equally important as reporting strong benchmark numbers.

---

## Motivation

Pedestrian detection is a foundational capability in autonomous vehicles, intelligent surveillance systems, and urban mobility planning. While detection performance on clean, well-separated pedestrians has matured significantly, occlusion remains a persistent and underappreciated challenge. When people walk in groups, stand in crowds, or are partially obscured, detection rates drop sharply — with real-world safety implications.

This project was designed to expose and quantify that gap, and to investigate whether data augmentation strategies can partially close it without access to larger crowd-specific datasets.

---

## Project Structure

```
pedestrian-detection-occlusion/
│
├── Pedestrian_Detection_Project.ipynb   # Full executed notebook with outputs
├── README.md                            # This file
├── img1.jpg                             # Real-world test images used for evaluation
├── img2.jpg                             # Real-world test images used for evaluation
├── img3.jpg                             # Real-world test images used for evaluation
```

> **Note:** Dataset files and model weights are not stored in this repository due to size constraints. Pre-trained model weights are available via the links below. The full dataset can be reproduced by running the notebook from Section 1.

---

## Pre-trained Model Weights

Both models are available for download and can be loaded directly with the Ultralytics YOLO API:

| Model | Description | Download |
|-------|-------------|----------|
| V1 Baseline | Fine-tuned on Penn-Fudan, no augmentation | [Download from Google Drive](https://drive.google.com/file/d/1t-7N6aD0zTyVy4e2oIQxN6e1dpyXPGBR/view?usp=sharing) |
| V2 Augmented | Retrained with occlusion + small-scale augmentation | [Download from Google Drive](https://drive.google.com/file/d/1XjNFG-Eb8gGSbUptiV6N0cR2JB2iwgjS/view?usp=sharing) |

To load a model:
```python
from ultralytics import YOLO

model = YOLO('v2_augmented.pt')
results = model('your_image.jpg', conf=0.3)
results[0].show()
```

---

## Dataset

**Penn-Fudan Pedestrian Dataset**
- 170 images, 423 annotated pedestrians
- Average of 2.5 pedestrians per image, up to 8 in a single frame
- Sourced from University of Pennsylvania and Fudan University campuses
- Annotation format: bounding boxes with pixel-level masks

> An important characteristic of this dataset is that severely occluded pedestrians are deliberately excluded from annotations. This is noted in the dataset's own readme and has direct implications for occlusion benchmarking, which is discussed in detail in the notebook.

---

## Methodology

### 1. Exploratory Data Analysis
The dataset was loaded, annotations were parsed from their native text format, and a statistical profile was built: pedestrian counts per image, bounding box size distributions, and visual inspection of occlusion cases. This stage identified the annotation policy of excluding severely occluded pedestrians — a finding that shaped how later evaluation results were interpreted.

### 2. Data Preprocessing
Annotations were converted from Penn-Fudan's native format to YOLO's normalised coordinate format (`class x_center y_center width height`), and the dataset was split into 136 training and 34 validation images using a fixed random seed for reproducibility.

### 3. Baseline Training (V1)
A YOLOv8n model pre-trained on COCO was fine-tuned on the pedestrian dataset for 50 epochs on a T4 GPU. Fine-tuning rather than training from scratch was chosen because the COCO pre-training already encodes strong priors about human body shapes, significantly reducing the data and compute required.

### 4. Occlusion Analysis
The validation set was partitioned into three occlusion buckets based on inter-pedestrian bounding box overlap (IoU):
- **Visible:** overlap < 10%
- **Partially occluded:** overlap 10–30%
- **Heavily occluded:** overlap > 30%

Recall was measured independently for each group. The analysis revealed a critical data imbalance: only 2 heavily occluded instances existed in the validation set, making the 100% recall figure for that category statistically unreliable. This finding was documented as a dataset limitation rather than a model strength.

### 5. Real-World Stress Testing
The V1 model was evaluated on three externally sourced images of genuinely crowded urban scenes — an aerial crosswalk, a New York City street crossing, and a Warsaw pedestrian crossing. Detection counts were compared against visual estimates of actual pedestrian counts. The model detected approximately 20–30% of visible pedestrians in dense crowds, exposing a significant domain gap between training and deployment conditions.

### 6. Data Augmentation and Retraining (V2)
Two custom augmentation strategies were implemented to address the identified failure modes:

**Occlusion patches:** Grey rectangles were randomly applied over 20–50% of each annotated pedestrian from one of four sides (top, bottom, left, right), simulating one person walking behind another.

**Small-scale copies:** Pedestrians were cropped, resized to 25–40% of their original size, and pasted into the upper (background) region of the image, along with corresponding label entries. This directly addresses the model's failure to detect distant pedestrians.

The original 136 training images were augmented to produce 544 training examples — a 4× increase without acquiring new data. The model was retrained from the same pre-trained YOLO weights for a fair comparison.

---

## Results

### Validation Set Performance

| Metric | V1 Baseline | V2 Augmented | Change |
|--------|-------------|--------------|--------|
| Precision | 0.895 | 0.950 | **+0.055** |
| Recall | 0.939 | 0.912 | -0.027 |
| mAP@50 | 0.958 | 0.961 | +0.003 |
| mAP@50-95 | 0.818 | 0.838 | **+0.020** |

Precision improved substantially, indicating the augmented model makes fewer false detections. The minor recall reduction reflects the standard precision-recall tradeoff. The mAP@50-95 improvement — the strictest metric, requiring tight box localisation across multiple IoU thresholds — suggests the augmented model produces more geometrically accurate detections.

### Occlusion Analysis

| Occlusion Level | Total Instances | Recall |
|-----------------|-----------------|--------|
| Visible | 72 | 90.3% |
| Partial | 17 | 88.2% |
| Heavy | 2 | 100%* |

*Statistically unreliable — based on 2 instances only.

### Real-World Crowd Detection

| Scene | V1 Detected | V2 Detected | Estimated Actual |
|-------|-------------|-------------|-----------------|
| Aerial crosswalk | 10 | 23 | 50+ |
| NYC street crossing | 11 | 14 | 50+ |
| Warsaw crosswalk | 13 | 14 | 40+ |

V2 demonstrated a 2.3× improvement on the hardest scene (aerial view), with more modest gains on eye-level scenes. Both models remain well below human-level counting ability in dense crowds.

[View my notebook](https://nbviewer.org/github/sojourner-so/pedestrian-detection-occlusion/blob/main/Pedestrian_Detection_Project.ipynb)

---

## Key Findings

**1. Clean benchmark metrics are not sufficient for occlusion evaluation.**
A mAP@50 of 0.958 on the validation set coexists with detecting fewer than 30% of pedestrians in real crowded scenes. Evaluation methodology must match the deployment context.

**2. Dataset quality bounds augmentation effectiveness.**
Augmenting a clean, small dataset produces meaningful but limited improvements. The domain gap between Penn-Fudan and genuine crowd scenes is structural — it cannot be fully closed through synthetic transformation of the existing data.

**3. Augmentation strategy should match observed failure modes.**
The small-scale augmentation produced the most visible real-world improvement (aerial scene: 10 → 23 detections) precisely because it targeted the specific failure mode of missing distant pedestrians — a direct translation from failure analysis to intervention design.

**4. Honest negative results are informative.**
The finding that severely occluded pedestrians are underrepresented in Penn-Fudan annotations is a methodological contribution in itself. Projects that report only strong numbers on curated validation sets risk overstating their systems' real-world readiness.

---

## Limitations and Future Work

The primary limitation of this project is the training dataset. Penn-Fudan was designed for ease of use and clean annotation, not for stress-testing occlusion handling. To meaningfully advance performance in crowded scenes, the following directions are recommended:

**Crowd-specific training data**
- [CrowdHuman](https://www.crowdhuman.org/) — 15,000 images with an average of 22 annotated pedestrians per image, including explicit occlusion labels
- [CityPersons](https://github.com/cvgroup-njust/CityPersons) — pedestrians extracted from urban driving footage across diverse cities and lighting conditions

**Architectural improvements**
- Larger YOLOv8 variants (YOLOv8m, YOLOv8l) offer better small-object detection at the cost of inference speed
- RT-DETR's transformer-based attention mechanism reasons about the full image context simultaneously, which is theoretically better suited to occlusion scenarios than sliding-window CNN approaches

**Evaluation rigour**
- Adopting the MR-FPPI (Miss Rate vs False Positives Per Image) metric used in the Caltech Pedestrian Benchmark would enable direct comparison with published research
- Testing across time-of-day and weather conditions would characterise robustness more completely

---

## Environment and Reproducibility

All experiments were conducted on Google Colab with a T4 GPU (15GB VRAM).

```
Python        3.12
ultralytics   8.4.18
torch         2.10.0+cu128
CUDA          12.8 (Tesla T4)
```

To reproduce:
```bash
pip install ultralytics filterpy lap
```

Then run the notebook cells in order. Training takes approximately 5–10 minutes per model on a T4 GPU. The dataset downloads automatically in Section 1.

---

## Acknowledgements

- **Penn-Fudan Pedestrian Dataset** — Liming Wang, Jianbo Shi, Gang Song, I-fan Shen. Object Detection Combining Recognition and Segmentation. ACCV 2007.
- **Ultralytics YOLOv8** — [ultralytics.com](https://ultralytics.com)

---

*This project was completed as a self-directed exploration of computer vision and pedestrian detection. All code, analysis, and documentation are original work.*
