# Open-Vocabulary USAR Victim Detection and Rescue Scene Understanding

An AI-assisted framework for **victim detection, open-vocabulary rescue landmark recognition, and scene understanding in Urban Search and Rescue (USAR) environments**.

The project combines a specialized **SimAM-FPN Faster R-CNN victim detector**, an **open-vocabulary object detection branch**, and a lightweight **Vision-Language Model (VLM)** to detect potential victims, identify rescue-relevant objects and hazards, construct semantic rescue landmarks, and generate structured operational reports.

The system is designed for research involving **earthquake environments, collapsed structures, UAV-assisted search and rescue, GPS-denied navigation, and disaster-scene perception**.

![](https://github.com/1Px-Vision/DisasterVision-SAR/blob/main/Victims_detection.jpg)

---

## Key Features

* Specialized **SimAM-FPN Faster R-CNN** victim detector
* Open-vocabulary rescue-scene detection
* Recognition of **`trapped person` as a victim candidate**
* Cross-model IoU comparison between:

  * SimAM-FPN victim detections
  * Open-vocabulary `trapped person` detections
* Optional ground-truth IoU evaluation
* Semantic rescue-landmark indexing
* Rescue-oriented object categorization
* Optional depth-based 3-D landmark projection
* VLM-based disaster-scene description
* Structured USAR operational report
* CUDA GPU acceleration
* 256 × 256 image inference
* Google Colab / Jupyter interactive interface
* PNG visualization
* JSON structured results
* TXT VLM scene reports

---

## System Architecture

The proposed framework contains three complementary perception branches:

```text
                         RGB / UAV Image
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
     SimAM-FPN Faster R-CNN          Open-Vocabulary Detector
       Specialized Victim               Rescue Vocabulary
           Detection                         │
               │                             │
               │                 ┌───────────┴───────────┐
               │                 │                       │
               │                 ▼                       ▼
               │          Trapped Person          Rescue Landmarks
               │          Victim Candidate      Hazards / Access /
               │                             Structures / Equipment
               │
               └───────────────┬───────────────────────┘
                               │
                               ▼
                       Cross-Model Fusion
                               │
                       IoU / Confidence
                               │
                               ▼
                    Rescue Landmark Index
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          2-D Semantic Map             Optional 3-D Map
                                      Depth + Intrinsics
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                            SmolVLM
                               │
                               ▼
                    Structured USAR Report
```

The specialized detector remains responsible for robust victim localization, while the open-vocabulary branch enables the system to recognize rescue concepts not necessarily included in the original victim-detector training dataset.

---

## Victim Detection

The primary detector is based on:

```text
Input Image
   ↓
Lightweight CNN Backbone
   ↓
SimAM Attention
   ↓
Feature Pyramid Network
   ↓
Region Proposal Network
   ↓
Faster R-CNN ROI Heads
   ↓
Victim Bounding Boxes
```

The detector uses two Faster R-CNN classes:

```text
0 → Background
1 → Victim
```

Predictions are filtered according to a configurable confidence threshold.

Example:

```text
SimAM-FPN V1 Victim
Confidence = 0.934
Box = [45.2, 71.8, 132.5, 226.1]
```

---

## Open-Vocabulary Rescue Detection

Unlike a conventional detector with a fixed number of classes, the open-vocabulary branch receives natural-language rescue concepts.

The default rescue vocabulary can include:

```python
RESCUE_VOCABULARY = [
    "person",
    "victim",
    "injured person",
    "trapped person",
    "unconscious person",
    "person lying on the ground",
    "rescue worker",

    "door",
    "doorway",
    "window",
    "stairs",
    "ladder",
    "corridor",
    "opening",
    "void space",

    "rubble",
    "debris",
    "collapsed wall",
    "concrete slab",
    "beam",
    "column",
    "damaged building",

    "fire",
    "smoke",
    "water",
    "electrical cable",
    "electrical panel",
    "gas cylinder",

    "helmet",
    "stretcher",
    "medical kit",
    "backpack",
]
```

The vocabulary can be modified without retraining the specialized Faster R-CNN detector.

---

## Trapped-Person Victim Detection

A particularly important feature is the interpretation of the open-vocabulary concept:

```text
"trapped person"
```

as a **Victim Detection** rather than only a generic semantic landmark.

Example:

```text
Victim Detection
trapped person
Confidence = 0.82
```

The system then compares the `trapped person` bounding box with victim boxes produced by SimAM-FPN Faster R-CNN.

---

## Cross-Model IoU

For a SimAM-FPN victim bounding box (B_S) and an open-vocabulary trapped-person box (B_T), spatial agreement is calculated as

$$
\operatorname{IoU}(B_S,B_T)
=
\frac{|B_S \cap B_T|}
{|B_S \cup B_T|}
$$

For example:

```text
Open-Vocabulary:
Victim detection
trapped person
Conf = 0.824
IoU(SimAM V1) = 0.713
```

and:

```text
SimAM-FPN:
SimAM-FPN V1 Victim
Conf = 0.914
IoU(trapped person) = 0.713
MATCH
```

A configurable IoU threshold can be used to classify the cross-model relationship:

```text
IoU ≥ 0.50 → MATCH
IoU < 0.50 → LOW_IOU
```

### Important

Cross-model IoU is an **agreement measure between two AI models**.

It should not be interpreted as ground-truth accuracy.

When human-annotated bounding boxes are available, the software reports ground-truth IoU separately.

---

## Ground-Truth Evaluation

Optional YOLO-format annotations can be supplied:

```text
class_id x_center y_center width height
```

All coordinates are normalized between 0 and 1.

Example:

```text
0 0.512 0.463 0.284 0.510
```

When annotations are available, the program calculates:

* True Positives
* False Positives
* False Negatives
* Precision
* Recall
* IoU
* Mean TP IoU

Example:

```text
Detected victims: 3
Ground-truth victims: 2

TP = 2
FP = 1
FN = 0

Precision = 0.667
Recall = 1.000
Mean TP IoU = 0.742
```

When ground truth is unavailable:

```text
IoU = N/A
```

This prevents model-to-model agreement from being confused with annotation-based evaluation.

---

## Rescue Landmark Indexing

Open-vocabulary detections are converted into rescue landmarks.

A landmark can contain:

```text
Landmark ID
Object / concept
Category
Confidence
Bounding box
Image center
Semantic grid position
Operational priority
Victim association
3-D position, when available
```

Conceptually,

[
L_k =
{
c_k,
s_k,
u_k,
v_k,
g_x,
g_y,
C_k,
P_k
},
]

where:

* (c_k): semantic concept
* (s_k): detection confidence
* (u_k,v_k): image coordinates
* (g_x,g_y): semantic-grid coordinates
* (C_k): rescue category
* (P_k): operational indexing priority

---

## Rescue Categories

Detected concepts are automatically grouped into operational categories such as:

```text
victim
responder
hazard
access
structure
equipment
landmark
```

For example:

| Detection        | Rescue Category |
| ---------------- | --------------- |
| trapped person   | victim          |
| injured person   | victim          |
| rescue worker    | responder       |
| smoke            | hazard          |
| electrical cable | hazard          |
| doorway          | access          |
| stairs           | access          |
| collapsed wall   | structure       |
| concrete slab    | structure       |
| stretcher        | equipment       |

These labels support higher-level semantic mapping rather than simple object detection.

---

## Vision-Language Model

The framework also integrates a lightweight VLM for USAR-oriented scene understanding.

Default configuration:

```python
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
```

The VLM receives information from both the image and detector context.

The generated operational report contains:

```text
1. SCENE SUMMARY
2. VISIBLE STRUCTURAL / DEBRIS CONDITIONS
3. VICTIM OBSERVATIONS
4. ACCESS AND EGRESS OBSERVATIONS
5. VISIBLE HAZARDS
6. USAR OPERATIONAL PRIORITIES
7. UNCERTAINTY / ITEMS REQUIRING HUMAN VERIFICATION
```

The VLM is instructed not to invent hazards, victims, injuries, structural conditions, or safe routes that cannot be visually supported.

---

## Example Processing Pipeline

```python
image
   ↓
resize 256 × 256
   ↓
SimAM-FPN Faster R-CNN
   ↓
candidate victim boxes
   │
   ├───────────────┐
   │               │
   ▼               ▼
Ground Truth     Open-Vocabulary Detector
IoU              ↓
                 trapped person
                 rubble
                 smoke
                 doorway
                 ...
                   │
                   ▼
          Cross-model victim IoU
                   │
                   ▼
          Rescue landmark indexing
                   │
                   ▼
                SmolVLM
                   │
                   ▼
         Structured USAR report
```

---

## Installation

The project is intended primarily for **Google Colab** or a CUDA-enabled Python environment.

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY>
```

### Install Dependencies

```bash
pip install -U \
    torch \
    torchvision \
    transformers \
    accelerate \
    safetensors \
    numpy \
    pillow \
    matplotlib \
    ipywidgets
```

For Google Colab:

```python
!pip install -q -U transformers accelerate safetensors ipywidgets
```

A separate `flash-attn` installation is not required by the supplied VLM configuration.

---

## Model Checkpoint

Place the trained SimAM-FPN Faster R-CNN model in:

```text
/content/SAR_SimAM_FasterRCNN/
```

Default model:

```text
fasterrcnn_simam_fpn_best.pt
```

Example configuration:

```python
MODEL_PT = Path(
    "/content/SAR_SimAM_FasterRCNN/"
    "fasterrcnn_simam_fpn_best.pt"
)
```

The checkpoint must correspond to the same SimAM-FPN Faster R-CNN architecture used by the inference program.

---

## Running the Project

Main program:

```text
sar_vlm_open_vocab_trapped_iou.py
```

Run inside Colab or Jupyter:

```python
%run sar_vlm_open_vocab_trapped_iou.py
```

or from a terminal:

```bash
python sar_vlm_open_vocab_trapped_iou.py
```

---

## Input Modes

The project supports two principal input modes.

### External Image

```python
TEST_MODE = "image"

IMAGE_PATH = Path(
    "/content/victims_8.jpg"
)
```

Supported formats include:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

### NPY Dataset

```python
TEST_MODE = "npy"

DATA_ROOT = Path(
    "/content/drive/MyDrive/SAR_Earthquake_NPY"
)

TEST_INDEX = 0
```

Expected files include:

```text
X_test.npy
y_test.npy
box_count_test.npy
```

---

## Ground-Truth Annotation

Ground truth is optional.

```python
GT_YOLO_PATH = None
```

or:

```python
GT_YOLO_PATH = Path(
    "/content/victims_8.txt"
)
```

If no annotation is supplied, victim detection continues normally but annotation-based IoU is reported as:

```text
N/A
```

---

## Important Parameters

### Victim confidence

```python
SCORE_THRESHOLD = 0.30
```

### Ground-truth IoU

```python
IOU_THRESHOLD = 0.50
```

### Maximum number of detections

```python
MAX_DETECTIONS = 50
```

### Detector input resolution

```python
INFERENCE_SIZE = 256
```

These values should be adjusted using a validation dataset rather than selected from a single example image.

---

## CUDA Acceleration

The program automatically checks for CUDA:

```python
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

When a GPU is available, inference runs on CUDA.

Example output:

```text
Device: cuda
GPU: NVIDIA T4
Detector loaded successfully.
```

For repeated inference, detector and VLM instances are cached to avoid unnecessary model reloading.

---

## Visualization

The generated figure distinguishes different information sources.

Typical visualization:

```text
Red solid box
→ SimAM-FPN victim prediction

Magenta dashed box
→ Open-vocabulary trapped-person victim prediction

Green box
→ Ground-truth victim, when available
```

Example annotation:

```text
SimAM-FPN V1 Victim
Conf = 0.914
IoU(GT) = 0.782
IoU(trapped person) = 0.713
MATCH
```

The visualization therefore separates:

```text
Detection confidence
Ground-truth IoU
Cross-model IoU
```

---

## Output Files

Typical outputs include:

```text
<sample>_victim_iou_plot.png

<sample>_victim_iou.json

<sample>_open_vocabulary_rescue_map.png

<sample>_rescue_landmarks.json

<sample>_USAR_VLM_report.txt
```

The precise set of files depends on the enabled modules.

---

## Example JSON Concept

A victim candidate may be represented as:

```json
{
  "victim_id": 1,
  "source": "SimAM-FPN",
  "confidence": 0.914,
  "box_xyxy": [
    48.2,
    66.4,
    137.9,
    224.3
  ],
  "ground_truth_iou": 0.782,
  "trapped_person_iou": 0.713,
  "cross_model_status": "MATCH"
}
```

An open-vocabulary victim candidate may appear as:

```json
{
  "label": "trapped person",
  "category": "victim",
  "confidence": 0.824,
  "box_xyxy": [
    51.7,
    69.1,
    140.4,
    221.8
  ],
  "best_simam_victim": 1,
  "cross_model_iou": 0.713
}
```

---

## Research Motivation

Victim detection in disaster environments is difficult because visual conditions can vary substantially because of:

* debris
* occlusion
* dust
* collapsed structures
* irregular victim poses
* partial visibility
* poor illumination
* unusual viewpoints
* UAV camera motion

A detector trained only with a fixed `victim` class may fail to represent the semantic diversity of rescue scenes.

The proposed framework therefore combines a **specialized trained detector** with **open-vocabulary semantic perception**.

This enables concepts such as:

```text
trapped person
person under rubble
injured person
person lying down
void space
collapsed wall
smoke
doorway
stretcher
```

to contribute to rescue-scene understanding without requiring every concept to be included as a dedicated Faster R-CNN training class.

---

## Proposed Research Contribution

The framework investigates a hybrid perception strategy:

[
\text{Specialized Detection}
+
\text{Open-Vocabulary Perception}
+
\text{Semantic Mapping}
+
\text{Vision-Language Reasoning}.
]

The specialized SimAM-FPN detector provides task-specific victim localization, while the open-vocabulary detector expands the semantic representation of the environment.

Cross-model IoU then provides an interpretable measure of spatial agreement:

[
A_{ij}
======

IoU
\left(
B_i^{SimAM},
B_j^{OV}
\right).
]

The resulting semantic landmarks can subsequently be associated with depth, UAV pose, VIO, SLAM, or UWB localization to construct a rescue-oriented spatial map.

---

## Potential UAV Integration

The framework can serve as the perception component of a GPS-denied rescue UAV:

```text
UAV RGB / RGB-D Camera
          │
          ▼
    Victim Detection
          │
          ▼
Open-Vocabulary Scene Perception
          │
          ▼
   Rescue Landmark Map
          │
     ┌────┴────┐
     ▼         ▼
    VIO       UWB
     │         │
     └────┬────┘
          ▼
   3-D Rescue Map
          │
          ▼
 Path Planning / USAR Team
```

Possible future extensions include:

* VIO-based landmark registration
* 3-D semantic SLAM
* UWB-based UAV localization
* multiple-drone semantic mapping
* temporal victim tracking
* thermal-camera fusion
* depth-camera fusion
* LiDAR fusion
* uncertainty-aware rescue prioritization
* real-time edge inference
* FPGA / embedded acceleration

---

## Limitations

This project is intended as **research and decision-support software**.

Several limitations should be considered:

* A detected person is not automatically a confirmed victim.
* `trapped person` is an AI semantic prediction and requires human verification.
* High cross-model IoU does not constitute ground truth.
* A single RGB image cannot determine structural stability.
* A VLM cannot reliably determine medical condition from appearance alone.
* Occlusion and debris can reduce victim-detection recall.
* Open-vocabulary scores are not necessarily calibrated probabilities.
* Scene descriptions may contain uncertainty and should be reviewed by trained personnel.
* The system should not independently authorize entry into hazardous structures.

---

## Responsible Use

The system is intended to assist trained personnel in:

* Urban Search and Rescue research
* UAV disaster-response experiments
* victim-localization studies
* rescue-scene semantic mapping
* perception benchmarking
* GPS-denied navigation research

AI predictions should be interpreted as **candidate observations requiring human verification**.

The framework must not be considered a replacement for trained emergency responders, structural engineers, medical personnel, or established USAR operational procedures.

---

## Project File

Current implementation:

```text
sar_vlm_open_vocab_trapped_iou.py
```

Core technologies:

```text
Python
PyTorch
TorchVision
SimAM
Feature Pyramid Network
Faster R-CNN
Open-Vocabulary Detection
Transformers
SmolVLM
Matplotlib
Google Colab
CUDA
```

---

## Suggested Repository Name

```text
RescueVision-OpenVocab-USAR
```

Alternative:

```text
OpenRescue-VLM
```

or:

```text
SAR-OpenVocab-VictimNet
```

---

## Suggested Citation

If this project contributes to academic work, please cite the corresponding publication or repository release associated with the implementation.

A repository citation can later be provided through a `CITATION.cff` file.

---

## License

Add the license appropriate for your research and model dependencies.

For example:

```text
MIT License
```

Before redistribution, verify the licenses and usage conditions of all pretrained models, datasets, and external dependencies included in the project.

---

## Acknowledgment

This project explores the integration of **specialized deep-learning victim detection, open-vocabulary visual perception, semantic rescue mapping, and vision-language reasoning** for AI-assisted Urban Search and Rescue applications.

The objective is to improve situational awareness in complex disaster environments while maintaining explicit uncertainty and human verification throughout the rescue decision-support pipeline.
