# SAR Victim Detection and USAR Scene Understanding

A **deep-learning framework for victim detection and environment understanding in Urban Search and Rescue (USAR) scenarios** using a lightweight **SimAM-FPN Faster R-CNN** detector combined with a **Vision-Language Model (VLM)**.

The application is designed for disaster-response imagery such as earthquake rubble, collapsed structures, damaged urban environments, and search-and-rescue scenes. It detects potential victims and generates a structured description of the surrounding environment to support situational awareness in USAR.

![](https://github.com/1Px-Vision/DisasterVision-SAR/blob/main/SAR_Victims.jpg)

---

## Overview

The proposed system combines two complementary AI components:

1. **SimAM-FPN Faster R-CNN**

   * Detects candidate victims in disaster imagery.
   * Uses a lightweight custom convolutional backbone.
   * Incorporates **SimAM attention**.
   * Uses a custom **Feature Pyramid Network (FPN)**.
   * Performs inference on **256 × 256 RGB images**.

2. **Vision-Language Model**

   * Default model: `HuggingFaceTB/SmolVLM-500M-Instruct`.
   * Analyzes the visual environment.
   * Receives victim detections as contextual information.
   * Generates an operationally oriented USAR scene description.

The detector output is treated as **candidate victim information**, not as verified ground truth.

---

## System Architecture

```text
                         RGB Disaster Image
                                │
                                ▼
                     Image Preprocessing
                         256 × 256 RGB
                                │
                                ▼
                 SimAM-FPN Faster R-CNN
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
             Victim Bounding Boxes    Confidence Scores
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                     Detector Context
                                │
                ┌───────────────┴───────────────┐
                │                               │
                │ Original RGB Scene            │
                │                               ▼
                └──────────────────────► Vision-Language Model
                                         SmolVLM-500M-Instruct
                                                  │
                                                  ▼
                                       USAR Scene Description
                                                  │
                     ┌────────────────────────────┼─────────────────────────┐
                     ▼                            ▼                         ▼
               Scene Summary                Visible Hazards        Victim Observations
                     │                            │                         │
                     ├────────────────────────────┼─────────────────────────┤
                     ▼                            ▼                         ▼
             Access / Egress              USAR Priorities          Uncertainty Report
```

---

## Main Features

* Victim detection in post-disaster imagery.
* SimAM attention mechanism.
* Lightweight multi-scale CNN backbone.
* Custom Feature Pyramid Network.
* Faster R-CNN object detector.
* Fixed **256 × 256 RGB detector input**.
* CUDA/GPU acceleration.
* Confidence-based victim filtering.
* Optional YOLO-format ground-truth annotations.
* IoU calculation.
* One-to-one predicted/ground-truth matching.
* TP, FP, and FN evaluation.
* Precision and recall calculation.
* Mean true-positive IoU.
* Vision-Language Model integration.
* Structured USAR environment description.
* Google Colab interactive interface.
* Image-upload mode.
* NPY dataset test mode.
* Google Drive support.
* PNG visualization export.
* JSON detection report.
* Text-based USAR VLM report.
* Model caching for repeated inference.

---

## USAR Scene Understanding

The VLM produces a structured operational report containing:

### 1. Scene Summary

General description of the observed disaster environment.

### 2. Visible Structural / Debris Conditions

Visual observations such as:

* collapsed building elements,
* rubble,
* concrete debris,
* damaged walls,
* obstructed areas,
* visible structural damage.

### 3. Victim Observations

Candidate victim information based on:

* Faster R-CNN detections,
* confidence values,
* approximate victim location within the image,
* visually observable context.

### 4. Access and Egress Observations

Potentially relevant visible information regarding:

* blocked passages,
* open areas,
* debris accumulation,
* apparent access paths,
* obstacles.

### 5. Visible Hazards

Examples may include visually observable:

* unstable-looking debris,
* large concrete fragments,
* damaged structures,
* blocked areas,
* exposed obstacles.

The VLM is instructed not to invent hazards that cannot be visually confirmed.

### 6. USAR Operational Priorities

The system can identify observations that may warrant human attention, such as:

* verifying detected victim locations,
* inspecting obstructed areas,
* assessing visible debris around victims,
* evaluating potential access routes.

### 7. Uncertainty / Human Verification

The system explicitly reports information that cannot be reliably determined from a single RGB image.

---

## Important Safety Principle

This application is intended as an **AI-assisted visual decision-support system**.

It must **not** be used as an autonomous authority for:

* structural safety assessment,
* declaring a building safe to enter,
* medical diagnosis,
* determining victim condition,
* confirming utility hazards,
* confirming safe rescue routes,
* replacing trained USAR personnel.

All AI-generated detections and VLM descriptions require validation by qualified rescue personnel.

---

## Detector Architecture

The detector uses the following processing chain:

```text
RGB Image
   │
   ▼
Conv Stem
   │
   ▼
Inverted Residual Blocks
   │
   ├── SimAM
   │
   ▼
Multi-scale CNN Features
   │
   ▼
Custom Lightweight FPN
   │
   ├── P2
   ├── P3
   ├── P4
   └── P5
   │
   ▼
Region Proposal Network
   │
   ▼
MultiScaleRoIAlign
   │
   ▼
Faster R-CNN Detection Head
   │
   ▼
Victim
```

The detector contains two classes internally:

```text
0 = background
1 = victim
```

---

## SimAM Attention

The backbone integrates the **Simple, Parameter-Free Attention Module (SimAM)**.

SimAM enhances feature responses without introducing a conventional trainable attention network.

It is applied within selected inverted residual blocks to improve the representation of relevant victim features in complex disaster scenes.

---

## Input Resolution

The detector operates using:

```text
256 × 256 × 3
```

where:

```text
Width    = 256 pixels
Height   = 256 pixels
Channels = RGB
```

External images are converted using:

```python
image = (
    ImageOps
    .exif_transpose(image)
    .convert("RGB")
    .resize(
        (256, 256),
        Image.Resampling.BILINEAR,
    )
)
```

---

## Vision-Language Model

The default VLM is:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

The implementation uses a compatibility loader supporting:

```python
AutoModelForImageTextToText
```

with fallback to:

```python
AutoModelForVision2Seq
```

for older `transformers` versions.

The model is loaded only when VLM analysis is requested.

---

## USAR VLM Prompt

The VLM is instructed to analyze only visible evidence and detector context.

The default report structure is:

```text
1. SCENE SUMMARY

2. VISIBLE STRUCTURAL / DEBRIS CONDITIONS

3. VICTIM OBSERVATIONS

4. ACCESS AND EGRESS OBSERVATIONS

5. VISIBLE HAZARDS

6. USAR OPERATIONAL PRIORITIES

7. UNCERTAINTY / ITEMS REQUIRING HUMAN VERIFICATION
```

The prompt explicitly instructs the model not to invent:

* people,
* injuries,
* hazards,
* utility conditions,
* building occupancy,
* structural stability,
* safe access routes.

---

## Google Colab Installation

Create a new Google Colab notebook and enable GPU acceleration:

```text
Runtime
   ↓
Change runtime type
   ↓
Hardware accelerator
   ↓
GPU
```

Install the required packages:

```bash
pip install -q -U transformers accelerate safetensors ipywidgets
```

TorchVision and PyTorch are normally preinstalled in Google Colab.

If required:

```bash
pip install torch torchvision
```

---

## FlashAttention

`flash-attn` is **not required** by the supplied implementation.

The VLM loader uses an attention configuration compatible with standard PyTorch execution.

This avoids long FlashAttention compilation times in Google Colab.

---

## Running the Application

Execute the Python code in Colab and launch:

```python
create_colab_interface()
```

The interactive interface will appear in the notebook.

---

## Google Colab Interface

The interface provides the following controls.

### Model Configuration

```text
Detector .pt
VLM model
Output folder
Compute device
```

### Input Modes

Two input modes are available.

#### Upload Image

Upload:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

Optional YOLO ground truth:

```text
.txt
```

#### NPY Dataset

The application can read:

```text
X_test.npy
y_test.npy
box_count_test.npy
```

and select a sample using its test index.

---

## Detection Controls

### Confidence Threshold

Example:

```text
0.30
```

Only predictions satisfying:

```text
confidence >= threshold
```

are retained.

### IoU Threshold

Example:

```text
0.50
```

A detection can be classified as a true positive when:

```text
IoU >= IoU threshold
```

and the corresponding ground-truth box has not already been assigned.

### Maximum Detections

Controls the maximum number of victim candidates returned for each image.

---

## Vision-Language Controls

The interface includes:

```text
Generate USAR VLM scene report
VLM model
Maximum generation tokens
Editable USAR prompt
```

The VLM can therefore be enabled or disabled independently from the object detector.

---

## IoU Evaluation

When ground truth is available, the application computes Intersection over Union:

```text
                 Area(Prediction ∩ Ground Truth)
IoU = ------------------------------------------------
                 Area(Prediction ∪ Ground Truth)
```

Example:

```text
IoU = 0.82
```

with:

```text
IoU threshold = 0.50
```

results in:

```text
True Positive
```

assuming the ground-truth victim has not already been matched.

---

## Ground-Truth Format

Ground-truth files use normalized YOLO format:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.273438 0.658203 0.179688 0.613281
0 0.431641 0.611328 0.191406 0.472656
```

where:

```text
0 = victim
```

and all coordinates are normalized to:

```text
0.0 – 1.0
```

---

## Example Detection Output

```text
================================================================
VICTIM DETECTION RESULTS
================================================================

Detected victims: 2
Ground-truth victims: 2

P1:
Victim
confidence = 0.882
IoU = 0.81
GT = 1
TP

P2:
Victim
confidence = 0.703
IoU = 0.74
GT = 2
TP
```

---

## Example USAR Report

An example VLM response may have the following structure:

```text
1. SCENE SUMMARY

The image shows a heavily damaged urban environment with extensive
concrete debris and partially collapsed structural elements.

2. VISIBLE STRUCTURAL / DEBRIS CONDITIONS

Large quantities of rubble and fragmented concrete are visible.
Several areas appear obstructed by debris.

3. VICTIM OBSERVATIONS

The detector reports two candidate victim locations in the lower-central
portion of the image. These detections require confirmation by USAR personnel.

4. ACCESS AND EGRESS OBSERVATIONS

Direct access to the candidate locations appears partially obstructed
by rubble. A safe access route cannot be confirmed from the image.

5. VISIBLE HAZARDS

Large debris and damaged structural elements are visible.
Structural stability is not visually confirmed.

6. USAR OPERATIONAL PRIORITIES

Verify the two candidate victim locations and conduct an on-site
structural and access assessment before entry.

7. UNCERTAINTY / ITEMS REQUIRING HUMAN VERIFICATION

Victim condition, structural stability, utility hazards, and safe
access routes cannot be confirmed from this image.
```

---

## Output Files

For an image such as:

```text
victims_8.jpg
```

the application generates:

```text
victims_8_victim_iou_plot.png
victims_8_victim_iou.json
victims_8_USAR_VLM_report.txt
```

---

## JSON Output

Example:

```json
{
  "sample": "victims_8",
  "score_threshold": 0.5,
  "iou_threshold": 0.5,
  "detected_victims": 2,
  "ground_truth_victims": 2,
  "tp": 2,
  "fp": 0,
  "fn": 0,
  "detections": [
    {
      "victim_id": 1,
      "confidence": 0.882,
      "box_xyxy": [
        48.2,
        91.1,
        134.7,
        232.4
      ],
      "best_iou": 0.81,
      "best_gt_id": 1,
      "status": "TP"
    }
  ],
  "usar_vlm": {
    "enabled": true,
    "model": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "generation_time_ms": 820.5,
    "report_file": "victims_8_USAR_VLM_report.txt",
    "report": "USAR scene description..."
  }
}
```

---

## Repository Structure

A suggested repository layout is:

```text
SAR-Victim-Detection-USAR-VLM/
│
├── README.md
│
├── LICENSE
│
├── requirements.txt
│
├── src/
│   └── SAR_FasterRCNN_SmolVLM_USAR_Colab_256_FIXED.py
│
├── models/
│   └── README.md
│
├── examples/
│   ├── victims_8.jpg
│   └── victims_8.txt
│
├── results/
│   ├── victim_iou_plot.png
│   ├── victim_iou.json
│   └── USAR_VLM_report.txt
│
└── notebooks/
    └── SAR_USAR_Colab.ipynb
```

Large `.pt` model files should normally not be committed directly to GitHub.

Consider using:

```text
Git LFS
```

or providing a separate model-download link.

---

## Requirements

Suggested `requirements.txt`:

```text
numpy
pillow
matplotlib
torch
torchvision
transformers
accelerate
safetensors
ipywidgets
```

---

## Model File

The Faster R-CNN checkpoint expected by the application is:

```text
fasterrcnn_simam_fpn_best.pt
```

Example Colab path:

```python
MODEL_PT = Path(
    "/content/SAR_SimAM_FasterRCNN/"
    "fasterrcnn_simam_fpn_best.pt"
)
```

---

## CUDA Support

The application automatically selects CUDA when available:

```python
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

On a GPU-enabled Colab runtime, the interface displays the detected NVIDIA GPU.

---

## Model Caching

Both AI models are cached after loading:

```text
Faster R-CNN
SmolVLM
```

Therefore, subsequent image analyses do not require model reloading unless the selected model changes.

This significantly reduces repeated inference overhead.

---

## Intended Applications

Potential research applications include:

* Urban Search and Rescue.
* Earthquake response.
* Post-disaster victim detection.
* UAV-based disaster reconnaissance.
* Robotic disaster exploration.
* Emergency-response situational awareness.
* AI-assisted SAR mapping.
* Human detection in rubble environments.
* Edge-assisted UAV perception.
* Multimodal rescue robotics.
* Disaster-response decision support.

---

## UAV Integration

The system can be extended to operate with UAV imagery:

```text
UAV Camera
    │
    ▼
RGB Frame
    │
    ▼
Resize 256 × 256
    │
    ▼
Victim Detector
    │
    ├── Victim position
    └── Confidence
    │
    ▼
VLM Environment Analysis
    │
    ▼
USAR Situation Report
    │
    ▼
Ground Control Station
```

Future versions may combine this system with:

* GPS-denied navigation,
* Visual-Inertial Odometry,
* SLAM,
* UWB localization,
* WiFi-CSI sensing,
* thermal imaging,
* LiDAR,
* RF sensing,
* multi-UAV coordination.

---

## Limitations

The current system has several important limitations.

### RGB Dependence

The VLM and detector operate primarily on RGB imagery.

Performance may decrease under:

* smoke,
* darkness,
* severe occlusion,
* dust,
* motion blur,
* extreme viewing angles.

### Victim Occlusion

Victims partially or completely covered by debris may not be detected.

### VLM Hallucination Risk

Vision-Language Models can produce incorrect or unsupported statements.

For this reason, the supplied prompt requires explicit uncertainty reporting and human verification.

### Structural Assessment

A monocular RGB image cannot provide reliable structural engineering assessment.

### Ground Truth

IoU, precision, recall, TP, FP, and FN require valid ground-truth annotations.

When ground truth is unavailable:

```text
IoU = N/A
```

---

## Future Work

Possible future improvements include:

* real-time UAV video processing,
* thermal-camera integration,
* RGB-thermal fusion,
* depth estimation,
* LiDAR fusion,
* UWB victim localization,
* RF-based victim sensing,
* WiFi-CSI respiration detection,
* multi-UAV cooperative perception,
* temporal VLM reasoning,
* VLM-guided UAV navigation,
* automatic disaster-scene mapping,
* georeferenced victim reporting,
* edge-device deployment,
* TensorRT acceleration,
* model quantization,
* ONNX export,
* FPGA-assisted inference,
* uncertainty calibration,
* rescue-priority estimation.

---

## Research Disclaimer

This project is intended for **research, experimentation, and decision-support development**.

The software should not be interpreted as a certified life-safety, structural-assessment, or autonomous rescue system.

All detections and scene descriptions must be reviewed by qualified personnel before operational decisions are made.

---

## Citation

If this repository contributes to academic research, please cite the corresponding publication or project documentation.

Example placeholder:

```bibtex
@software{sar_usar_vlm,
  title  = {SAR Victim Detection and USAR Scene Understanding
            Using SimAM-FPN Faster R-CNN and Vision-Language Models},
  author = {Carlos Osorio Quero},
  year   = {2026},
  note   = {Research software for AI-assisted Urban Search and Rescue}
}
```

Update the citation information when the associated publication is available.

---

## License

Select a license according to the intended use of the project.

Common options include:

```text
MIT License
Apache License 2.0
BSD 3-Clause License
```

Verify that the selected license is compatible with the licenses of the pretrained models and datasets used by the project.

---

## Acknowledgments

This project builds on open-source technologies including:

* PyTorch
* TorchVision
* Hugging Face Transformers
* SmolVLM
* NumPy
* Pillow
* Matplotlib
* Google Colab

The framework is developed as a research platform for **AI-assisted victim detection and situational awareness in Urban Search and Rescue environments**.
