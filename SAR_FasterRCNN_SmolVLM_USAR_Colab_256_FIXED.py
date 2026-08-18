import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageOps

# Google Colab / Jupyter interface
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    widgets = None
    display = None
    clear_output = None
    IPYWIDGETS_AVAILABLE = False

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou


# ============================================================
# GOOGLE COLAB DEPENDENCIES
# ============================================================
# Run once if needed:
#
# !pip install -q -U transformers accelerate safetensors ipywidgets
#
# The VLM is lazy-loaded only when requested.
#
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PT = Path("/content/SAR_SimAM_FasterRCNN/fasterrcnn_simam_fpn_best.pt")

DATA_ROOT = Path(
    "/content/drive/MyDrive/"
    "SAR_Earthquake_NPY"
)

OUTPUT_DIR = Path("/content/SAR_SimAM_FasterRCNN")

# ------------------------------------------------------------
# Select test mode:
#
# "image" = read IMAGE_PATH
# "npy"   = read one X_test.npy sample
# ------------------------------------------------------------

TEST_MODE = "image"

# ------------------------------------------------------------
# EXTERNAL IMAGE MODE
# ------------------------------------------------------------

IMAGE_PATH = Path(
    "/content/victims_8.jpg"
)

# Optional ground truth.
# Set None when unavailable.
GT_YOLO_PATH = None

# Example:
# GT_YOLO_PATH = Path(
#     "/content/victims_28.txt"
# )

# ------------------------------------------------------------
# NPY TEST MODE
# ------------------------------------------------------------

TEST_INDEX = 0

# ------------------------------------------------------------
# Detection thresholds
# ------------------------------------------------------------

SCORE_THRESHOLD = 0.30

IOU_THRESHOLD = 0.50

MAX_DETECTIONS = 50

# Fixed inference resolution requested for the SAR detector.
INFERENCE_SIZE = 256

# ------------------------------------------------------------
# Vision-Language Model (VLM) for USAR environment description
# ------------------------------------------------------------
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
VLM_MAX_NEW_TOKENS = 220

USAR_VLM_PROMPT = """
You are a visual decision-support assistant for an Urban Search and Rescue
(USAR) team. Analyze only what is visibly supported by the image and by the
detector context supplied below.

Produce a concise operational scene report with these headings:

1. SCENE SUMMARY
2. VISIBLE STRUCTURAL / DEBRIS CONDITIONS
3. VICTIM OBSERVATIONS
4. ACCESS AND EGRESS OBSERVATIONS
5. VISIBLE HAZARDS
6. USAR OPERATIONAL PRIORITIES
7. UNCERTAINTY / ITEMS REQUIRING HUMAN VERIFICATION

Rules:
- Do not invent people, injuries, hazards, utilities, structural stability,
  building occupancy, or safe routes that are not visibly supported.
- Treat Faster R-CNN detections as candidate victim locations, not ground truth.
- Never declare a structure safe to enter from an image alone.
- If a detail cannot be determined, say "not visually confirmed".
- Keep recommendations observational and verification-oriented.
- Use clear language suitable for a USAR team leader.
""".strip()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GOOGLE DRIVE
# ============================================================

def mount_drive_if_colab():
    try:
        import google.colab  # noqa
        from google.colab import drive

        if not Path(
            "/content/drive/MyDrive"
        ).exists():

            drive.mount(
                "/content/drive"
            )

    except ImportError:
        pass


# ============================================================
# SIMAM
# ============================================================

class SimAM(nn.Module):

    def __init__(
        self,
        e_lambda=1e-4,
    ):
        super().__init__()

        self.e_lambda = (
            e_lambda
        )

    def forward(
        self,
        x,
    ):
        n = max(
            x.shape[-1]
            * x.shape[-2]
            - 1,
            1,
        )

        mean = x.mean(
            dim=(2, 3),
            keepdim=True,
        )

        d = (
            x - mean
        ).pow(2)

        variance = (
            d.sum(
                dim=(2, 3),
                keepdim=True,
            )
            / n
        )

        energy = (
            d
            /
            (
                4.0
                * (
                    variance
                    + self.e_lambda
                )
            )
            + 0.5
        )

        return (
            x
            * torch.sigmoid(
                energy
            )
        )


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

class ConvBNAct(
    nn.Sequential
):

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        groups=1,
        activation=True,
    ):
        padding = (
            kernel_size
            // 2
        )

        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_ch
            ),
        ]

        if activation:
            layers.append(
                nn.Hardswish(
                    inplace=True
                )
            )

        super().__init__(
            *layers
        )


# ============================================================
# INVERTED RESIDUAL + SIMAM
# ============================================================

class InvertedResidualSimAM(
    nn.Module
):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride,
        expand_ratio,
        use_simam,
    ):
        super().__init__()

        hidden_ch = int(
            round(
                in_ch
                * expand_ratio
            )
        )

        self.use_residual = (
            stride == 1
            and in_ch == out_ch
        )

        layers = []

        if hidden_ch != in_ch:

            layers.append(
                ConvBNAct(
                    in_ch,
                    hidden_ch,
                    kernel_size=1,
                )
            )

        layers.append(
            ConvBNAct(
                hidden_ch,
                hidden_ch,
                kernel_size=3,
                stride=stride,
                groups=hidden_ch,
            )
        )

        if use_simam:

            layers.append(
                SimAM()
            )

        layers.append(
            ConvBNAct(
                hidden_ch,
                out_ch,
                kernel_size=1,
                activation=False,
            )
        )

        self.block = (
            nn.Sequential(
                *layers
            )
        )

    def forward(
        self,
        x,
    ):
        y = self.block(
            x
        )

        if self.use_residual:
            y = y + x

        return y


# ============================================================
# SAR BACKBONE
# ============================================================

class SARBackbone(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.stem = ConvBNAct(
            3,
            16,
            3,
            stride=2,
        )

        self.stage2 = nn.Sequential(
            InvertedResidualSimAM(
                16,
                24,
                2,
                4.0,
                False,
            ),
            InvertedResidualSimAM(
                24,
                24,
                1,
                3.0,
                False,
            ),
        )

        self.stage3 = nn.Sequential(
            InvertedResidualSimAM(
                24,
                40,
                2,
                4.0,
                True,
            ),
            InvertedResidualSimAM(
                40,
                40,
                1,
                3.0,
                True,
            ),
            InvertedResidualSimAM(
                40,
                40,
                1,
                3.0,
                True,
            ),
        )

        self.stage4 = nn.Sequential(
            InvertedResidualSimAM(
                40,
                80,
                2,
                4.0,
                True,
            ),
            InvertedResidualSimAM(
                80,
                80,
                1,
                3.0,
                True,
            ),
            InvertedResidualSimAM(
                80,
                96,
                1,
                3.0,
                True,
            ),
        )

        self.stage5 = nn.Sequential(
            InvertedResidualSimAM(
                96,
                160,
                2,
                4.0,
                True,
            ),
            InvertedResidualSimAM(
                160,
                160,
                1,
                3.0,
                True,
            ),
        )

        self.out_channels = [
            24,
            40,
            96,
            160,
        ]

    def forward(
        self,
        x,
    ):
        x = self.stem(
            x
        )

        c2 = self.stage2(
            x
        )

        c3 = self.stage3(
            c2
        )

        c4 = self.stage4(
            c3
        )

        c5 = self.stage5(
            c4
        )

        return [
            c2,
            c3,
            c4,
            c5,
        ]


# ============================================================
# CUSTOM FPN
# ============================================================

class LightweightFPN(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.lateral = (
            nn.ModuleList(
                [
                    nn.Conv2d(
                        c,
                        out_channels,
                        kernel_size=1,
                    )
                    for c
                    in in_channels
                ]
            )
        )

        self.smooth = (
            nn.ModuleList(
                [
                    ConvBNAct(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                    )
                    for _
                    in in_channels
                ]
            )
        )

    def forward(
        self,
        features,
    ):
        lateral = [
            conv(feature)
            for conv, feature
            in zip(
                self.lateral,
                features,
            )
        ]

        pyramid = [
            None
        ] * len(
            lateral
        )

        pyramid[-1] = (
            lateral[-1]
        )

        for i in range(
            len(lateral) - 2,
            -1,
            -1,
        ):
            up = (
                nn.functional
                .interpolate(
                    pyramid[
                        i + 1
                    ],
                    size=(
                        lateral[i]
                        .shape[-2:]
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            pyramid[i] = (
                lateral[i]
                + up
            )

        return [
            smooth(p)
            for smooth, p
            in zip(
                self.smooth,
                pyramid,
            )
        ]


# ============================================================
# FASTER R-CNN BACKBONE ADAPTER
# ============================================================

class SARSimAMFPNDetectionBackbone(
    nn.Module
):

    def __init__(
        self,
        fpn_channels=160,
    ):
        super().__init__()

        self.backbone = (
            SARBackbone()
        )

        self.fpn = (
            LightweightFPN(
                self.backbone
                .out_channels,
                fpn_channels,
            )
        )

        # Required by TorchVision FasterRCNN.
        self.out_channels = (
            fpn_channels
        )

    def forward(
        self,
        x,
    ):
        features = (
            self.backbone(
                x
            )
        )

        pyramid = (
            self.fpn(
                features
            )
        )

        return OrderedDict(
            (
                str(i),
                p,
            )
            for i, p
            in enumerate(
                pyramid
            )
        )


# ============================================================
# BUILD SAME DETECTOR AS TRAINING
# ============================================================

def build_detector(
    image_size,
    fpn_channels,
):
    backbone = (
        SARSimAMFPNDetectionBackbone(
            fpn_channels=(
                fpn_channels
            )
        )
    )

    anchor_generator = (
        AnchorGenerator(
            sizes=(
                (16, 24, 32),
                (32, 48, 64),
                (64, 96, 128),
                (128, 192, 256),
            ),
            aspect_ratios=(
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
            ),
        )
    )

    roi_pooler = (
        MultiScaleRoIAlign(
            featmap_names=[
                "0",
                "1",
                "2",
                "3",
            ],
            output_size=7,
            sampling_ratio=2,
        )
    )

    model = FasterRCNN(
        backbone=backbone,

        num_classes=2,

        min_size=image_size,
        max_size=image_size,

        rpn_anchor_generator=(
            anchor_generator
        ),

        box_roi_pool=(
            roi_pooler
        ),

        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,

        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,

        box_score_thresh=0.05,
        box_nms_thresh=0.50,

        box_detections_per_img=100,
    )

    return model


# ============================================================
# LOAD .PT MODEL
# ============================================================

def load_detector(
    model_path,
):
    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Detector .pt not found:\n"
            f"{model_path}"
        )

    print(
        "=" * 72
    )

    print(
        "LOADING SAR SimAM-FPN "
        "FASTER R-CNN"
    )

    print(
        "=" * 72
    )

    print(
        model_path
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Expected dictionary checkpoint."
        )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    architecture = (
        checkpoint.get(
            "architecture",
            "unknown",
        )
    )

    print(
        "Architecture:",
        architecture,
    )

    # Protect against accidentally loading
    # simam_fpn_classifier_best.pt.
    if (
        "fasterrcnn"
        not in str(
            architecture
        ).lower()
    ):

        state_keys = (
            checkpoint[
                "model_state_dict"
            ].keys()
        )

        if not any(
            k.startswith(
                "roi_heads."
            )
            for k in state_keys
        ):

            raise RuntimeError(
                "\nThe selected .pt file is "
                "not the adapted Faster R-CNN "
                "detector.\n\n"
                "Use:\n"
                "fasterrcnn_simam_fpn_best.pt\n"
                "instead of:\n"
                "simam_fpn_classifier_best.pt"
            )

    # Force the TorchVision detector transform to the same resolution
    # used by the Colab interface. Faster R-CNN weights are unaffected
    # because min_size/max_size are transform parameters, not learned weights.
    image_size = INFERENCE_SIZE

    fpn_channels = int(
        checkpoint.get(
            "fpn_channels",
            160,
        )
    )

    model = build_detector(
        image_size=(
            image_size
        ),
        fpn_channels=(
            fpn_channels
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Image size:",
        image_size,
    )

    print(
        "FPN channels:",
        fpn_channels,
    )

    print(
        "Device:",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda
            .get_device_name(
                0
            ),
        )

    print(
        "Detector loaded successfully."
    )

    return (
        model,
        checkpoint,
    )


# ============================================================
# IMAGE -> TORCH TENSOR
# ============================================================

def image_to_tensor(
    image,
):
    array = np.asarray(
        image.convert(
            "RGB"
        ),
        dtype=np.uint8,
    ).copy()

    tensor = (
        torch
        .from_numpy(
            array
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        .div(
            255.0
        )
    )

    return tensor


# ============================================================
# NORMALIZED XYWH -> ABSOLUTE XYXY
# ============================================================

def normalized_xywh_to_xyxy(
    rows,
    width,
    height,
):
    rows = np.asarray(
        rows,
        dtype=np.float32,
    )

    if rows.size == 0:

        return torch.empty(
            (0, 4),
            dtype=torch.float32,
        )

    rows = rows.reshape(
        -1,
        5,
    )

    xc = (
        rows[:, 1]
        * width
    )

    yc = (
        rows[:, 2]
        * height
    )

    bw = (
        rows[:, 3]
        * width
    )

    bh = (
        rows[:, 4]
        * height
    )

    x1 = (
        xc
        - bw / 2.0
    )

    y1 = (
        yc
        - bh / 2.0
    )

    x2 = (
        xc
        + bw / 2.0
    )

    y2 = (
        yc
        + bh / 2.0
    )

    boxes = np.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        axis=1,
    )

    boxes[:, [0, 2]] = (
        np.clip(
            boxes[:, [0, 2]],
            0,
            width - 1,
        )
    )

    boxes[:, [1, 3]] = (
        np.clip(
            boxes[:, [1, 3]],
            0,
            height - 1,
        )
    )

    valid = (
        (
            boxes[:, 2]
            - boxes[:, 0]
        )
        > 0
    ) & (
        (
            boxes[:, 3]
            - boxes[:, 1]
        )
        > 0
    )

    return torch.from_numpy(
        boxes[
            valid
        ].astype(
            np.float32
        )
    )


# ============================================================
# LOAD YOLO GROUND TRUTH
# ============================================================

def load_yolo_gt(
    label_path,
    width,
    height,
):
    if label_path is None:

        return torch.empty(
            (0, 4),
            dtype=torch.float32,
        )

    label_path = Path(
        label_path
    )

    if not label_path.exists():

        raise FileNotFoundError(
            f"GT annotation not found:\n"
            f"{label_path}"
        )

    rows = []

    for line in (
        label_path
        .read_text(
            encoding="utf-8",
            errors="ignore",
        )
        .splitlines()
    ):
        parts = (
            line.strip()
            .split()
        )

        if len(parts) < 5:
            continue

        try:
            row = [
                float(
                    parts[i]
                )
                for i
                in range(5)
            ]

        except ValueError:
            continue

        rows.append(
            row
        )

    return (
        normalized_xywh_to_xyxy(
            rows,
            width,
            height,
        )
    )


# ============================================================
# LOAD NPY TEST IMAGE + GT
# ============================================================

def load_npy_test(
    root,
    index,
):
    root = Path(
        root
    )

    X = np.load(
        root
        / "X_test.npy",
        mmap_mode="r",
    )

    Y = np.load(
        root
        / "y_test.npy",
        mmap_mode="r",
    )

    counts = np.load(
        root
        / "box_count_test.npy",
        mmap_mode="r",
    )

    if len(X) == 0:

        raise RuntimeError(
            "X_test.npy is empty."
        )

    index = (
        int(index)
        % len(X)
    )

    image_array = np.array(
        X[index],
        dtype=np.uint8,
        copy=True,
    )

    image = (
        Image.fromarray(
            image_array
        )
        .convert(
            "RGB"
        )
    )

    n = int(
        counts[index]
    )

    rows = np.asarray(
        Y[
            index,
            :n,
        ],
        dtype=np.float32,
    )

    gt_boxes = (
        normalized_xywh_to_xyxy(
            rows,
            image.width,
            image.height,
        )
    )

    return (
        image,
        gt_boxes,
        index,
    )


# ============================================================
# DETECT VICTIMS
# ============================================================

@torch.inference_mode()
def detect_victims(
    model,
    image,
):
    tensor = (
        image_to_tensor(
            image
        )
        .to(
            DEVICE
        )
    )

    output = model(
        [
            tensor
        ]
    )[0]

    boxes = (
        output[
            "boxes"
        ]
        .detach()
        .cpu()
    )

    labels = (
        output[
            "labels"
        ]
        .detach()
        .cpu()
    )

    scores = (
        output[
            "scores"
        ]
        .detach()
        .cpu()
    )

    # Class 1 = victim.
    keep = (
        (
            labels
            == 1
        )
        &
        (
            scores
            >= SCORE_THRESHOLD
        )
    )

    boxes = (
        boxes[
            keep
        ]
    )

    scores = (
        scores[
            keep
        ]
    )

    if len(
        scores
    ) > 0:

        order = torch.argsort(
            scores,
            descending=True,
        )

        boxes = (
            boxes[
                order
            ]
        )

        scores = (
            scores[
                order
            ]
        )

    return (
        boxes[
            :MAX_DETECTIONS
        ],
        scores[
            :MAX_DETECTIONS
        ],
    )


# ============================================================
# IoU + ONE-TO-ONE MATCHING
# ============================================================

def calculate_iou_matches(
    pred_boxes,
    gt_boxes,
):
    num_pred = len(
        pred_boxes
    )

    num_gt = len(
        gt_boxes
    )

    if num_pred == 0:

        return {
            "best_iou": (
                np.zeros(
                    0,
                    dtype=np.float32,
                )
            ),
            "best_gt": (
                np.zeros(
                    0,
                    dtype=np.int64,
                )
            ),
            "status": [],
            "tp": 0,
            "fp": 0,
            "fn": (
                num_gt
            ),
        }

    if num_gt == 0:

        return {
            "best_iou": (
                np.zeros(
                    num_pred,
                    dtype=np.float32,
                )
            ),
            "best_gt": (
                np.full(
                    num_pred,
                    -1,
                    dtype=np.int64,
                )
            ),
            "status": [
                "N/A"
            ] * num_pred,
            "tp": 0,
            "fp": (
                num_pred
            ),
            "fn": 0,
        }

    matrix = box_iou(
        pred_boxes,
        gt_boxes,
    )

    best_values, best_indices = (
        matrix.max(
            dim=1
        )
    )

    matched_gt = set()

    status = []

    tp = 0
    fp = 0

    # Predictions are already ordered
    # by confidence.
    for pred_id in range(
        num_pred
    ):
        values, indices = (
            torch.sort(
                matrix[
                    pred_id
                ],
                descending=True,
            )
        )

        matched = False

        for value, gt_id in zip(
            values.tolist(),
            indices.tolist(),
        ):
            if gt_id in matched_gt:
                continue

            if value >= IOU_THRESHOLD:

                matched_gt.add(
                    gt_id
                )

                matched = True
                tp += 1

            break

        if matched:
            status.append(
                "TP"
            )

        else:
            status.append(
                "FP"
            )

            fp += 1

    fn = (
        num_gt
        - len(
            matched_gt
        )
    )

    return {
        "best_iou": (
            best_values
            .numpy()
        ),
        "best_gt": (
            best_indices
            .numpy()
        ),
        "status": status,
        "tp": int(
            tp
        ),
        "fp": int(
            fp
        ),
        "fn": int(
            fn
        ),
    }


# ============================================================
# PLOT RESULT WITH MATPLOTLIB
# ============================================================

def plot_detection(
    image,
    pred_boxes,
    pred_scores,
    gt_boxes,
    match_info,
    sample_name,
):
    fig, ax = plt.subplots(
        figsize=(
            13,
            9,
        )
    )

    ax.imshow(
        image
    )

    ax.axis(
        "off"
    )

    ax.set_title(
        (
            "SAR Victim Detection — "
            "SimAM-FPN Faster R-CNN\n"
            f"Confidence ≥ "
            f"{SCORE_THRESHOLD:.2f}"
            f" | IoU TP ≥ "
            f"{IOU_THRESHOLD:.2f}"
        ),
        fontsize=15,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Ground truth boxes
    # --------------------------------------------------------

    for gt_id, box in enumerate(
        gt_boxes.tolist(),
        start=1,
    ):
        x1, y1, x2, y2 = (
            box
        )

        rect = (
            patches.Rectangle(
                (
                    x1,
                    y1,
                ),
                x2 - x1,
                y2 - y1,
                linewidth=2.5,
                edgecolor="lime",
                facecolor="none",
            )
        )

        ax.add_patch(
            rect
        )

        ax.text(
            x1,
            max(
                2,
                y1 - 5,
            ),
            f"GT{gt_id} Victim",
            fontsize=10,
            color="black",
            bbox=dict(
                facecolor="lime",
                alpha=0.85,
                pad=2,
            ),
        )

    # --------------------------------------------------------
    # Predicted victims
    # --------------------------------------------------------

    for pred_id, (
        box,
        score,
    ) in enumerate(
        zip(
            pred_boxes.tolist(),
            pred_scores.tolist(),
        ),
        start=1,
    ):
        x1, y1, x2, y2 = (
            box
        )

        rect = (
            patches.Rectangle(
                (
                    x1,
                    y1,
                ),
                x2 - x1,
                y2 - y1,
                linewidth=2.5,
                edgecolor="red",
                facecolor="none",
            )
        )

        ax.add_patch(
            rect
        )

        if len(
            gt_boxes
        ) > 0:

            iou = float(
                match_info[
                    "best_iou"
                ][
                    pred_id - 1
                ]
            )

            gt_index = (
                int(
                    match_info[
                        "best_gt"
                    ][
                        pred_id - 1
                    ]
                )
                + 1
            )

            status = (
                match_info[
                    "status"
                ][
                    pred_id - 1
                ]
            )

            text = (
                f"P{pred_id} Victim\n"
                f"Conf={score:.3f}\n"
                f"IoU={iou:.3f} "
                f"with GT{gt_index}\n"
                f"{status}"
            )

        else:

            text = (
                f"P{pred_id} Victim\n"
                f"Conf={score:.3f}\n"
                "IoU=N/A"
            )

        ax.text(
            x1,
            min(
                image.height - 10,
                y2 + 4,
            ),
            text,
            fontsize=9,
            color="white",
            bbox=dict(
                facecolor="red",
                alpha=0.75,
                pad=3,
            ),
        )

    # --------------------------------------------------------
    # Summary panel
    # --------------------------------------------------------

    n_pred = len(
        pred_boxes
    )

    n_gt = len(
        gt_boxes
    )

    if n_gt > 0:

        tp = (
            match_info[
                "tp"
            ]
        )

        fp = (
            match_info[
                "fp"
            ]
        )

        fn = (
            match_info[
                "fn"
            ]
        )

        precision = (
            tp
            / max(
                tp + fp,
                1,
            )
        )

        recall = (
            tp
            / max(
                tp + fn,
                1,
            )
        )

        tp_ious = [
            float(
                match_info[
                    "best_iou"
                ][i]
            )
            for i, status
            in enumerate(
                match_info[
                    "status"
                ]
            )
            if status
            == "TP"
        ]

        mean_iou = (
            float(
                np.mean(
                    tp_ious
                )
            )
            if tp_ious
            else 0.0
        )

        summary = (
            f"Detected victims: {n_pred}\n"
            f"Ground truth: {n_gt}\n"
            f"TP={tp} | FP={fp} | FN={fn}\n"
            f"Precision={precision:.3f}\n"
            f"Recall={recall:.3f}\n"
            f"Mean TP IoU={mean_iou:.3f}"
        )

    else:

        summary = (
            f"Detected victims: {n_pred}\n"
            "Ground truth unavailable\n"
            "IoU = N/A"
        )

    ax.text(
        0.015,
        0.985,
        summary,
        transform=(
            ax.transAxes
        ),
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.72,
            boxstyle="round,pad=0.5",
        ),
    )

    # Legend.
    legend_items = [
        patches.Patch(
            facecolor="none",
            edgecolor="red",
            label=(
                "Predicted victim"
            ),
        )
    ]

    if len(
        gt_boxes
    ) > 0:

        legend_items.append(
            patches.Patch(
                facecolor="none",
                edgecolor="lime",
                label=(
                    "Ground-truth victim"
                ),
            )
        )

    ax.legend(
        handles=(
            legend_items
        ),
        loc="lower right",
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_plot = (
        OUTPUT_DIR
        / (
            f"{sample_name}_"
            "victim_iou_plot.png"
        )
    )

    fig.savefig(
        output_plot,
        dpi=180,
        bbox_inches="tight",
    )

    # Important for Colab:
    plt.show()

    plt.close(
        fig
    )

    return output_plot


# ============================================================
# PRINT DETECTION TABLE
# ============================================================

def print_results(
    pred_boxes,
    pred_scores,
    gt_boxes,
    match_info,
):
    print(
        "\n"
        + "=" * 72
    )

    print(
        "VICTIM DETECTION RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        "Detected victims:",
        len(
            pred_boxes
        ),
    )

    print(
        "Ground-truth victims:",
        len(
            gt_boxes
        ),
    )

    if len(
        pred_boxes
    ) == 0:

        print(
            "No victim detection "
            "above threshold."
        )

        return

    for i in range(
        len(
            pred_boxes
        )
    ):
        score = float(
            pred_scores[i]
        )

        box = [
            round(
                float(v),
                1,
            )
            for v in (
                pred_boxes[i]
                .tolist()
            )
        ]

        if len(
            gt_boxes
        ) > 0:

            iou = float(
                match_info[
                    "best_iou"
                ][i]
            )

            gt_id = (
                int(
                    match_info[
                        "best_gt"
                    ][i]
                )
                + 1
            )

            status = (
                match_info[
                    "status"
                ][i]
            )

            print(
                f"P{i+1}: "
                f"Victim | "
                f"confidence={score:.4f} | "
                f"IoU={iou:.4f} | "
                f"GT={gt_id} | "
                f"{status} | "
                f"box={box}"
            )

        else:

            print(
                f"P{i+1}: "
                f"Victim | "
                f"confidence={score:.4f} | "
                f"IoU=N/A | "
                f"box={box}"
            )


# ============================================================
# SAVE JSON
# ============================================================

def save_json_report(
    sample_name,
    pred_boxes,
    pred_scores,
    gt_boxes,
    match_info,
    output_plot,
):
    report = {
        "sample": (
            sample_name
        ),
        "model": str(
            MODEL_PT
        ),
        "score_threshold": float(
            SCORE_THRESHOLD
        ),
        "iou_threshold": float(
            IOU_THRESHOLD
        ),
        "ground_truth_victims": int(
            len(
                gt_boxes
            )
        ),
        "detected_victims": int(
            len(
                pred_boxes
            )
        ),
        "tp": int(
            match_info[
                "tp"
            ]
        ),
        "fp": int(
            match_info[
                "fp"
            ]
        ),
        "fn": int(
            match_info[
                "fn"
            ]
        ),
        "plot": str(
            output_plot
        ),
        "detections": [],
    }

    for i in range(
        len(
            pred_boxes
        )
    ):
        item = {
            "victim_id": int(
                i + 1
            ),
            "confidence": float(
                pred_scores[i]
            ),
            "box_xyxy": [
                float(v)
                for v in (
                    pred_boxes[i]
                    .tolist()
                )
            ],
        }

        if len(
            gt_boxes
        ) > 0:

            item[
                "best_iou"
            ] = float(
                match_info[
                    "best_iou"
                ][i]
            )

            item[
                "best_gt_id"
            ] = (
                int(
                    match_info[
                        "best_gt"
                    ][i]
                )
                + 1
            )

            item[
                "status"
            ] = (
                match_info[
                    "status"
                ][i]
            )

        else:

            item[
                "best_iou"
            ] = None

            item[
                "best_gt_id"
            ] = None

            item[
                "status"
            ] = "N/A"

        report[
            "detections"
        ].append(
            item
        )

    output_json = (
        OUTPUT_DIR
        / (
            f"{sample_name}_"
            "victim_iou.json"
        )
    )

    output_json.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_json


# ============================================================
# MAIN
# ============================================================

def main_SAR_2():

    mount_drive_if_colab()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Read detector .pt
    # --------------------------------------------------------

    model, checkpoint = (
        load_detector(
            MODEL_PT
        )
    )

    # --------------------------------------------------------
    # 2. Read test image
    # --------------------------------------------------------

    if (
        TEST_MODE.lower()
        == "npy"
    ):

        (
            image,
            gt_boxes,
            resolved_index,
        ) = load_npy_test(
            DATA_ROOT,
            TEST_INDEX,
        )

        sample_name = (
            f"npy_test_"
            f"{resolved_index:05d}"
        )

        print(
            "\nReading NPY test "
            f"image {resolved_index}"
        )

    elif (
        TEST_MODE.lower()
        == "image"
    ):

        if not (
            IMAGE_PATH.exists()
        ):

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{IMAGE_PATH}"
            )

        image = (
            ImageOps
          .exif_transpose(
            Image.open(
              IMAGE_PATH
            )
          )
          .convert(
              "RGB"
          )
        .resize(
            (INFERENCE_SIZE, INFERENCE_SIZE),
            Image.Resampling.BILINEAR
        )
            )

        gt_boxes = (
            load_yolo_gt(
                GT_YOLO_PATH,
                image.width,
                image.height,
            )
        )

        sample_name = (
            IMAGE_PATH.stem
        )

        print(
            "\nReading image:"
        )

        print(
            IMAGE_PATH
        )

    else:

        raise ValueError(
            "TEST_MODE must be "
            "'image' or 'npy'."
        )

    print(
        "Image size:",
        image.size,
    )

    # --------------------------------------------------------
    # 3. Faster R-CNN inference
    # --------------------------------------------------------

    pred_boxes, pred_scores = (
        detect_victims(
            model,
            image,
        )
    )

    # --------------------------------------------------------
    # 4. IoU comparison
    # --------------------------------------------------------

    match_info = (
        calculate_iou_matches(
            pred_boxes,
            gt_boxes,
        )
    )

    # --------------------------------------------------------
    # 5. Print victim detections
    # --------------------------------------------------------

    print_results(
        pred_boxes,
        pred_scores,
        gt_boxes,
        match_info,
    )

    # --------------------------------------------------------
    # 6. Show detection plot using plt
    # --------------------------------------------------------

    output_plot = (
        plot_detection(
            image,
            pred_boxes,
            pred_scores,
            gt_boxes,
            match_info,
            sample_name,
        )
    )

    # --------------------------------------------------------
    # 7. Save result JSON
    # --------------------------------------------------------

    output_json = (
        save_json_report(
            sample_name,
            pred_boxes,
            pred_scores,
            gt_boxes,
            match_info,
            output_plot,
        )
    )

    print(
        "\nSaved plot:"
    )

    print(
        output_plot
    )

    print(
        "\nSaved JSON:"
    )

    print(
        output_json
    )
# ============================================================
# GOOGLE COLAB INTERACTIVE INTERFACE
# ============================================================

_UI_MODEL = None
_UI_CHECKPOINT = None
_UI_MODEL_PATH = None

_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_MODEL_ID_LOADED = None
_VLM_DEVICE_LOADED = None


def _uploaded_file(upload_widget):
    """
    Return (filename, bytes) from an ipywidgets.FileUpload object.

    Supports the tuple/list representation used by recent ipywidgets
    and the dictionary representation used by older Colab versions.
    """
    value = upload_widget.value

    if not value:
        return None, None

    if isinstance(value, dict):
        filename = next(iter(value))
        item = value[filename]
        content = item.get("content", b"")
        if hasattr(content, "tobytes"):
            content = content.tobytes()
        return filename, bytes(content)

    item = value[0]

    if isinstance(item, dict):
        filename = item.get("name", "uploaded_file")
        content = item.get("content", b"")
    else:
        filename = getattr(item, "name", "uploaded_file")
        content = getattr(item, "content", b"")

    if hasattr(content, "tobytes"):
        content = content.tobytes()

    return filename, bytes(content)


def _save_upload(upload_widget, destination_dir):
    filename, content = _uploaded_file(upload_widget)

    if filename is None:
        return None

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    output_path = destination_dir / safe_name
    output_path.write_bytes(content)

    return output_path


def _resize_image_256(image):
    """
    Convert a PIL image to RGB and force exactly 256 x 256.
    """
    return (
        ImageOps
        .exif_transpose(image)
        .convert("RGB")
        .resize(
            (INFERENCE_SIZE, INFERENCE_SIZE),
            Image.Resampling.BILINEAR,
        )
    )


def _scale_xyxy_boxes(boxes, old_w, old_h, new_w, new_h):
    """
    Scale absolute xyxy boxes when an NPY sample is resized.
    """
    if len(boxes) == 0:
        return boxes

    out = boxes.clone().float()

    sx = float(new_w) / max(float(old_w), 1.0)
    sy = float(new_h) / max(float(old_h), 1.0)

    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy

    out[:, [0, 2]] = out[:, [0, 2]].clamp(
        0,
        max(new_w - 1, 0),
    )

    out[:, [1, 3]] = out[:, [1, 3]].clamp(
        0,
        max(new_h - 1, 0),
    )

    return out


def _load_model_for_ui(model_path):
    """
    Cache the loaded detector so repeated detections do not reload the .pt file.
    """
    global _UI_MODEL, _UI_CHECKPOINT, _UI_MODEL_PATH
    global MODEL_PT

    model_path = Path(model_path)

    if (
        _UI_MODEL is not None
        and _UI_MODEL_PATH == str(model_path)
    ):
        return _UI_MODEL, _UI_CHECKPOINT

    MODEL_PT = model_path

    _UI_MODEL, _UI_CHECKPOINT = load_detector(
        model_path
    )

    _UI_MODEL_PATH = str(model_path)

    return _UI_MODEL, _UI_CHECKPOINT



def _vlm_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _load_vlm_for_ui(model_id=VLM_MODEL_ID):
    """
    Robust SmolVLM loader for Google Colab.

    Compatibility:
      - New Transformers: AutoModelForImageTextToText
      - Older Transformers: AutoModelForVision2Seq fallback
      - No flash-attn required
      - CUDA BF16 when supported, otherwise FP16
      - CPU FP32 fallback
    """
    global _VLM_MODEL
    global _VLM_PROCESSOR
    global _VLM_MODEL_ID_LOADED
    global _VLM_DEVICE_LOADED

    model_id = str(model_id).strip()

    if not model_id:
        raise ValueError("VLM model ID cannot be empty.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if (
        _VLM_MODEL is not None
        and _VLM_PROCESSOR is not None
        and _VLM_MODEL_ID_LOADED == model_id
        and _VLM_DEVICE_LOADED == str(device)
    ):
        print("[VLM] Using cached model.")
        return _VLM_MODEL, _VLM_PROCESSOR

    # ---------------------------------------------------------
    # Import Transformers with new API first.
    # ---------------------------------------------------------
    try:
        import transformers
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
            ModelClass = AutoModelForImageTextToText
            model_api = "AutoModelForImageTextToText"
        except ImportError:
            # Backward compatibility with older Transformers.
            from transformers import AutoModelForVision2Seq
            ModelClass = AutoModelForVision2Seq
            model_api = "AutoModelForVision2Seq"

    except Exception as exc:
        raise RuntimeError(
            "Could not import the Hugging Face VLM classes.\n\n"
            "Run this in a NEW Colab cell:\n"
            "!pip install -q -U transformers accelerate safetensors\n\n"
            "Then restart the runtime and execute the notebook again.\n\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    print("=" * 72)
    print("LOADING USAR VISION-LANGUAGE MODEL")
    print("=" * 72)
    print("Model       :", model_id)
    print("Transformers:", transformers.__version__)
    print("Loader API  :", model_api)
    print("Device      :", device)

    # ---------------------------------------------------------
    # Select safe dtype for the active Colab hardware.
    # ---------------------------------------------------------
    if device.type == "cuda":
        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print("Model dtype :", model_dtype)

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=False,
        )

        # -----------------------------------------------------
        # New Transformers commonly uses `dtype=`.
        # Older releases may require `torch_dtype=`.
        # Use eager attention to avoid flash-attn dependency.
        # -----------------------------------------------------
        try:
            model = ModelClass.from_pretrained(
                model_id,
                dtype=model_dtype,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )

        except TypeError as first_exc:
            print(
                "[VLM] New dtype API not accepted; "
                "retrying with torch_dtype..."
            )

            model = ModelClass.from_pretrained(
                model_id,
                torch_dtype=model_dtype,
                _attn_implementation="eager",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )

        model = model.to(device)
        model.eval()

    except Exception as exc:
        # Release partially allocated CUDA memory before reporting error.
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        raise RuntimeError(
            "\nVLM MODEL LOAD FAILED\n"
            "------------------------------------------------------------\n"
            f"Model: {model_id}\n"
            f"Transformers: {transformers.__version__}\n"
            f"Loader: {model_api}\n"
            f"Device: {device}\n"
            f"Dtype: {model_dtype}\n\n"
            "Recommended Colab repair:\n"
            "1. !pip uninstall -y transformers tokenizers\n"
            "2. !pip install -q -U transformers accelerate safetensors\n"
            "3. Runtime -> Restart session\n"
            "4. Run all cells again\n\n"
            "Do NOT install flash-attn for this version of the code.\n\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    _VLM_MODEL = model
    _VLM_PROCESSOR = processor
    _VLM_MODEL_ID_LOADED = model_id
    _VLM_DEVICE_LOADED = str(device)

    print("[VLM] Loaded successfully.")

    return model, processor


def print_vlm_environment():
    """
    Print Colab environment information useful for VLM troubleshooting.
    """
    print("=" * 72)
    print("VLM ENVIRONMENT")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "CUDA capability:",
            torch.cuda.get_device_capability(0),
        )
        print(
            "BF16 supported:",
            (
                torch.cuda.is_bf16_supported()
                if hasattr(torch.cuda, "is_bf16_supported")
                else "unknown"
            ),
        )

    try:
        import transformers
        print("Transformers:", transformers.__version__)
    except Exception as exc:
        print("Transformers import ERROR:", exc)

    try:
        import accelerate
        print("Accelerate:", accelerate.__version__)
    except Exception as exc:
        print("Accelerate import ERROR:", exc)

    print("VLM model:", VLM_MODEL_ID)


def _build_detector_context(
    pred_boxes,
    pred_scores,
    image_width,
    image_height,
):
    """
    Convert Faster R-CNN candidate detections to compact VLM context.
    """
    count = len(pred_boxes)

    if count == 0:
        return (
            "Faster R-CNN detector context: "
            "0 candidate victim detections above the selected confidence "
            "threshold. This does not prove that no victims are present."
        )

    lines = [
        (
            "Faster R-CNN detector context: "
            f"{count} candidate victim detection(s). "
            "These are model predictions, not ground truth."
        )
    ]

    for i, (box, score) in enumerate(
        zip(
            pred_boxes.tolist(),
            pred_scores.tolist(),
        ),
        start=1,
    ):
        x1, y1, x2, y2 = [
            float(v) for v in box
        ]

        cx = (
            ((x1 + x2) / 2.0)
            / max(float(image_width), 1.0)
        )

        cy = (
            ((y1 + y2) / 2.0)
            / max(float(image_height), 1.0)
        )

        horizontal = (
            "left"
            if cx < 0.33
            else "right"
            if cx > 0.67
            else "center"
        )

        vertical = (
            "upper"
            if cy < 0.33
            else "lower"
            if cy > 0.67
            else "middle"
        )

        lines.append(
            f"- Candidate P{i}: confidence={float(score):.3f}, "
            f"approximate image region={vertical}-{horizontal}, "
            f"box_xyxy=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]."
        )

    return "\n".join(lines)


@torch.inference_mode()
def describe_usar_environment(
    image,
    pred_boxes,
    pred_scores,
    model_id=VLM_MODEL_ID,
    user_prompt=None,
    max_new_tokens=VLM_MAX_NEW_TOKENS,
):
    """
    Generate a USAR-oriented visual scene report.
    """
    model, processor = _load_vlm_for_ui(
        model_id
    )

    device = _vlm_device()

    detector_context = _build_detector_context(
        pred_boxes,
        pred_scores,
        image.width,
        image.height,
    )

    base_prompt = (
        user_prompt.strip()
        if user_prompt is not None
        and user_prompt.strip()
        else USAR_VLM_PROMPT
    )

    complete_prompt = (
        base_prompt
        + "\n\n"
        + detector_context
        + "\n\n"
        + "Inspect the image and prepare the USAR scene report."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": complete_prompt,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image.convert("RGB")],
        return_tensors="pt",
    )

    moved_inputs = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved_inputs[key] = value.to(device)
        else:
            moved_inputs[key] = value

    generated_ids = model.generate(
        **moved_inputs,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
    )

    input_len = moved_inputs["input_ids"].shape[1]

    new_token_ids = generated_ids[
        :,
        input_len:,
    ]

    report_text = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )[0].strip()

    if not report_text:
        report_text = (
            "VLM returned an empty report. "
            "Human visual assessment is required."
        )

    return report_text


def save_usar_vlm_report(
    sample_name,
    report_text,
    model_id,
    detector_count,
    inference_ms=None,
):
    """
    Save the VLM scene description as a text report.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / f"{sample_name}_USAR_VLM_report.txt"
    )

    header = [
        "USAR VLM SCENE REPORT",
        "=" * 72,
        f"Sample: {sample_name}",
        f"VLM: {model_id}",
        f"Detector candidates: {int(detector_count)}",
    ]

    if inference_ms is not None:
        header.append(
            f"VLM generation time: {float(inference_ms):.2f} ms"
        )

    header.extend(
        [
            "",
            (
                "NOTE: AI-generated visual decision support. "
                "Human USAR verification is required."
            ),
            "=" * 72,
            "",
        ]
    )

    path.write_text(
        "\n".join(header)
        + report_text
        + "\n",
        encoding="utf-8",
    )

    return path


def add_usar_vlm_to_json(
    json_path,
    report_text,
    model_id,
    inference_ms,
    report_path,
):
    """
    Add VLM output to the existing Faster R-CNN JSON report.
    """
    json_path = Path(json_path)

    data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    data["usar_vlm"] = {
        "enabled": True,
        "model": str(model_id),
        "generation_time_ms": (
            float(inference_ms)
            if inference_ms is not None
            else None
        ),
        "report_file": str(report_path),
        "report": report_text,
        "warning": (
            "AI-generated visual decision support. "
            "Scene hazards, structural stability, victim condition, "
            "and route safety require human verification."
        ),
    }

    json_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    return json_path


def create_colab_interface():
    """
    Create the SAR Faster R-CNN Google Colab interface.

    Features
    --------
    - Mount Google Drive.
    - Select detector .pt path.
    - Upload one external RGB image.
    - Optional YOLO ground-truth .txt upload.
    - Or select a sample from X_test.npy.
    - Fixed 256 x 256 inference.
    - Confidence, IoU and max-detection controls.
    - CUDA status display.
    - Matplotlib detection result.
    - IoU / TP / FP / FN report.
    - SmolVLM USAR environment description.
    - Structured USAR scene report.
    - Saved PNG + JSON + USAR VLM TXT outputs.
    """
    global SCORE_THRESHOLD
    global IOU_THRESHOLD
    global MAX_DETECTIONS
    global DATA_ROOT
    global OUTPUT_DIR

    if not IPYWIDGETS_AVAILABLE:
        raise RuntimeError(
            "ipywidgets is unavailable. In Colab run:\n"
            "!pip install -q ipywidgets"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    title = widgets.HTML(
        value="""
        <div style="
            border:1px solid #d0d7de;
            border-radius:12px;
            padding:16px;
            margin-bottom:10px;
        ">
          <h2 style="margin:0;">SAR Victim Detection</h2>
          <div style="margin-top:5px;">
            SimAM-FPN Faster R-CNN + SmolVLM · Google Colab Interface
          </div>
          <div style="margin-top:5px;">
            Detector input: <b>256 × 256 RGB</b> ·
            VLM: <b>USAR environment description</b>
          </div>
        </div>
        """
    )

    device_text = (
        f"CUDA: {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "CPU"
    )

    device_label = widgets.HTML(
        value=f"<b>Compute device:</b> {device_text}"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    model_path_box = widgets.Text(
        value=str(MODEL_PT),
        description="Model .pt:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "110px"},
    )

    data_root_box = widgets.Text(
        value=str(DATA_ROOT),
        description="NPY folder:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "110px"},
    )

    output_dir_box = widgets.Text(
        value=str(OUTPUT_DIR),
        description="Output:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "110px"},
    )

    vlm_model_box = widgets.Text(
        value=VLM_MODEL_ID,
        description="VLM model:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "110px"},
    )

    # --------------------------------------------------------
    # Input mode
    # --------------------------------------------------------
    mode_dropdown = widgets.Dropdown(
        options=[
            ("Upload image", "image"),
            ("NPY test sample", "npy"),
        ],
        value="image",
        description="Input:",
        style={"description_width": "110px"},
    )

    image_upload = widgets.FileUpload(
        accept=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff",
        multiple=False,
        description="Upload image",
    )

    gt_upload = widgets.FileUpload(
        accept=".txt",
        multiple=False,
        description="YOLO GT (.txt)",
    )

    test_index_widget = widgets.BoundedIntText(
        value=0,
        min=0,
        max=10_000_000,
        step=1,
        description="Test index:",
        style={"description_width": "110px"},
    )

    # --------------------------------------------------------
    # Detection settings
    # --------------------------------------------------------
    score_slider = widgets.FloatSlider(
        value=float(SCORE_THRESHOLD),
        min=0.01,
        max=0.99,
        step=0.01,
        description="Confidence:",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="520px"),
    )

    iou_slider = widgets.FloatSlider(
        value=float(IOU_THRESHOLD),
        min=0.10,
        max=0.95,
        step=0.05,
        description="IoU TP:",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="520px"),
    )

    max_det_widget = widgets.IntSlider(
        value=int(MAX_DETECTIONS),
        min=1,
        max=100,
        step=1,
        description="Max victims:",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="520px"),
    )

    vlm_enabled_checkbox = widgets.Checkbox(
        value=True,
        description="Generate USAR VLM scene report",
        indent=False,
    )

    vlm_tokens_widget = widgets.IntSlider(
        value=int(VLM_MAX_NEW_TOKENS),
        min=64,
        max=512,
        step=16,
        description="VLM tokens:",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width="520px"),
    )

    usar_prompt_box = widgets.Textarea(
        value=USAR_VLM_PROMPT,
        description="USAR prompt:",
        layout=widgets.Layout(
            width="95%",
            height="260px",
        ),
        style={"description_width": "110px"},
    )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------
    mount_button = widgets.Button(
        description="Mount Drive",
        button_style="info",
        icon="folder-open",
    )

    load_button = widgets.Button(
        description="Load Detector",
        button_style="warning",
        icon="download",
    )

    load_vlm_button = widgets.Button(
        description="Load VLM",
        button_style="info",
        icon="eye",
    )

    run_button = widgets.Button(
        description="Run Detection",
        button_style="success",
        icon="play",
    )

    clear_button = widgets.Button(
        description="Clear Output",
        icon="trash",
    )

    status = widgets.HTML(
        value="<b>Status:</b> Ready"
    )

    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #d0d7de",
            padding="8px",
            width="100%",
        )
    )

    # --------------------------------------------------------
    # Dynamic visibility
    # --------------------------------------------------------
    def update_mode_visibility(*_):
        is_image = mode_dropdown.value == "image"

        image_upload.layout.display = (
            "" if is_image else "none"
        )
        gt_upload.layout.display = (
            "" if is_image else "none"
        )
        data_root_box.layout.display = (
            "none" if is_image else ""
        )
        test_index_widget.layout.display = (
            "none" if is_image else ""
        )

    mode_dropdown.observe(
        update_mode_visibility,
        names="value",
    )

    update_mode_visibility()

    # --------------------------------------------------------
    # Mount Drive callback
    # --------------------------------------------------------
    def on_mount_clicked(_):
        with output:
            clear_output(wait=True)

            try:
                from google.colab import drive

                print("Mounting Google Drive...")
                drive.mount(
                    "/content/drive",
                    force_remount=False,
                )

                status.value = (
                    "<b>Status:</b> Google Drive mounted."
                )

            except Exception as exc:
                status.value = (
                    "<b>Status:</b> Drive mount failed."
                )
                print(
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------
    # Load model callback
    # --------------------------------------------------------
    def on_load_clicked(_):
        with output:
            clear_output(wait=True)

            try:
                status.value = (
                    "<b>Status:</b> Loading model..."
                )

                _load_model_for_ui(
                    model_path_box.value.strip()
                )

                status.value = (
                    "<b>Status:</b> Model loaded successfully."
                )

            except Exception as exc:
                status.value = (
                    "<b>Status:</b> Model load failed."
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------
    # Load VLM callback
    # --------------------------------------------------------
    def on_load_vlm_clicked(_):
        with output:
            clear_output(wait=True)

            try:
                status.value = (
                    "<b>Status:</b> Loading VLM..."
                )

                _load_vlm_for_ui(
                    vlm_model_box.value.strip()
                )

                status.value = (
                    "<b>Status:</b> VLM loaded successfully."
                )

            except Exception as exc:
                status.value = (
                    "<b>Status:</b> VLM load failed."
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------
    # Run inference callback
    # --------------------------------------------------------
    def on_run_clicked(_):
        global SCORE_THRESHOLD
        global IOU_THRESHOLD
        global MAX_DETECTIONS
        global DATA_ROOT
        global OUTPUT_DIR
        global MODEL_PT

        run_button.disabled = True

        with output:
            clear_output(wait=True)

            try:
                # Apply current controls.
                SCORE_THRESHOLD = float(
                    score_slider.value
                )

                IOU_THRESHOLD = float(
                    iou_slider.value
                )

                MAX_DETECTIONS = int(
                    max_det_widget.value
                )

                DATA_ROOT = Path(
                    data_root_box.value.strip()
                )

                OUTPUT_DIR = Path(
                    output_dir_box.value.strip()
                )

                MODEL_PT = Path(
                    model_path_box.value.strip()
                )

                OUTPUT_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                status.value = (
                    "<b>Status:</b> Running inference..."
                )

                model, checkpoint = (
                    _load_model_for_ui(
                        MODEL_PT
                    )
                )

                # --------------------------------------------
                # Read input
                # --------------------------------------------
                if mode_dropdown.value == "image":

                    uploaded_image_path = (
                        _save_upload(
                            image_upload,
                            "/content/sar_ui_uploads",
                        )
                    )

                    if uploaded_image_path is None:
                        raise ValueError(
                            "Upload an image before running detection."
                        )

                    with Image.open(
                        uploaded_image_path
                    ) as pil_image:

                        vlm_image = (
                            ImageOps
                            .exif_transpose(
                                pil_image
                            )
                            .convert("RGB")
                            .copy()
                        )

                        image = _resize_image_256(
                            vlm_image
                        )

                    gt_path = _save_upload(
                        gt_upload,
                        "/content/sar_ui_uploads",
                    )

                    gt_boxes = load_yolo_gt(
                        gt_path,
                        image.width,
                        image.height,
                    )

                    sample_name = (
                        uploaded_image_path.stem
                    )

                    source_description = str(
                        uploaded_image_path
                    )

                else:
                    (
                        image,
                        gt_boxes,
                        resolved_index,
                    ) = load_npy_test(
                        DATA_ROOT,
                        int(
                            test_index_widget.value
                        ),
                    )

                    old_w = image.width
                    old_h = image.height

                    if (
                        old_w != INFERENCE_SIZE
                        or old_h != INFERENCE_SIZE
                    ):
                        image = _resize_image_256(
                            image
                        )

                        gt_boxes = _scale_xyxy_boxes(
                            gt_boxes,
                            old_w,
                            old_h,
                            INFERENCE_SIZE,
                            INFERENCE_SIZE,
                        )

                    sample_name = (
                        f"npy_test_{resolved_index:05d}"
                    )

                    source_description = (
                        f"{DATA_ROOT}/X_test.npy"
                        f" [index={resolved_index}]"
                    )

                    vlm_image = image.copy()

                print("=" * 72)
                print("SAR VICTIM DETECTION")
                print("=" * 72)
                print("Source       :", source_description)
                print("Model        :", MODEL_PT)
                print("Image size   :", image.size)
                print("Confidence   :", SCORE_THRESHOLD)
                print("IoU threshold:", IOU_THRESHOLD)
                print("Device       :", DEVICE)

                if DEVICE.type == "cuda":
                    print(
                        "GPU          :",
                        torch.cuda.get_device_name(0),
                    )

                print()

                # Show input preview.
                display(
                    image.resize(
                        (384, 384),
                        Image.Resampling.NEAREST,
                    )
                )

                # --------------------------------------------
                # Inference
                # --------------------------------------------
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                import time
                t0 = time.perf_counter()

                pred_boxes, pred_scores = (
                    detect_victims(
                        model,
                        image,
                    )
                )

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                inference_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                match_info = (
                    calculate_iou_matches(
                        pred_boxes,
                        gt_boxes,
                    )
                )

                print_results(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                    match_info,
                )

                print(
                    f"\nInference time: "
                    f"{inference_ms:.2f} ms"
                )

                if inference_ms > 0:
                    print(
                        f"Approx. inference FPS: "
                        f"{1000.0 / inference_ms:.2f}"
                    )

                # --------------------------------------------
                # USAR VLM scene understanding
                # --------------------------------------------
                vlm_report = None
                vlm_inference_ms = None
                vlm_report_path = None

                if vlm_enabled_checkbox.value:
                    status.value = (
                        "<b>Status:</b> Generating USAR VLM report..."
                    )

                    print(
                        "\n"
                        + "=" * 72
                    )
                    print(
                        "USAR VLM ENVIRONMENT DESCRIPTION"
                    )
                    print(
                        "=" * 72
                    )

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                    vlm_t0 = time.perf_counter()

                    vlm_report = describe_usar_environment(
                        image=vlm_image,
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        model_id=(
                            vlm_model_box.value.strip()
                        ),
                        user_prompt=(
                            usar_prompt_box.value
                        ),
                        max_new_tokens=(
                            vlm_tokens_widget.value
                        ),
                    )

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                    vlm_inference_ms = (
                        time.perf_counter()
                        - vlm_t0
                    ) * 1000.0

                    print(vlm_report)

                    print(
                        f"\nVLM generation time: "
                        f"{vlm_inference_ms:.2f} ms"
                    )

                    vlm_report_path = (
                        save_usar_vlm_report(
                            sample_name=sample_name,
                            report_text=vlm_report,
                            model_id=(
                                vlm_model_box.value.strip()
                            ),
                            detector_count=(
                                len(pred_boxes)
                            ),
                            inference_ms=(
                                vlm_inference_ms
                            ),
                        )
                    )

                # --------------------------------------------
                # Plot
                # --------------------------------------------
                output_plot = (
                    plot_detection(
                        image,
                        pred_boxes,
                        pred_scores,
                        gt_boxes,
                        match_info,
                        sample_name,
                    )
                )

                # --------------------------------------------
                # JSON report
                # --------------------------------------------
                output_json = (
                    save_json_report(
                        sample_name,
                        pred_boxes,
                        pred_scores,
                        gt_boxes,
                        match_info,
                        output_plot,
                    )
                )

                if vlm_report is not None:
                    add_usar_vlm_to_json(
                        json_path=output_json,
                        report_text=vlm_report,
                        model_id=(
                            vlm_model_box.value.strip()
                        ),
                        inference_ms=(
                            vlm_inference_ms
                        ),
                        report_path=(
                            vlm_report_path
                        ),
                    )

                print("\nSaved:")
                print("Plot :", output_plot)
                print("JSON :", output_json)

                if vlm_report_path is not None:
                    print(
                        "USAR :", vlm_report_path
                    )

                status.value = (
                    "<b>Status:</b> "
                    + (
                        "Detection + USAR VLM completed."
                        if vlm_report is not None
                        else "Detection completed."
                    )
                )

            except Exception as exc:
                status.value = (
                    "<b>Status:</b> Detection failed."
                )

                print("\nERROR")
                print("-" * 72)
                print(
                    f"{type(exc).__name__}: {exc}"
                )

                import traceback
                traceback.print_exc()

            finally:
                run_button.disabled = False

    def on_clear_clicked(_):
        output.clear_output()
        status.value = (
            "<b>Status:</b> Ready"
        )

    mount_button.on_click(
        on_mount_clicked
    )

    load_button.on_click(
        on_load_clicked
    )

    load_vlm_button.on_click(
        on_load_vlm_clicked
    )

    run_button.on_click(
        on_run_clicked
    )

    clear_button.on_click(
        on_clear_clicked
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    path_panel = widgets.VBox(
        [
            widgets.HTML("<h4>1. Model and output paths</h4>"),
            model_path_box,
            vlm_model_box,
            output_dir_box,
            device_label,
        ]
    )

    input_panel = widgets.VBox(
        [
            widgets.HTML("<h4>2. Test input</h4>"),
            mode_dropdown,
            image_upload,
            gt_upload,
            data_root_box,
            test_index_widget,
        ]
    )

    threshold_panel = widgets.VBox(
        [
            widgets.HTML("<h4>3. Detection settings</h4>"),
            score_slider,
            iou_slider,
            max_det_widget,
        ]
    )

    vlm_panel = widgets.VBox(
        [
            widgets.HTML(
                "<h4>4. USAR Vision-Language Model</h4>"
            ),
            widgets.HTML(
                "<div>"
                "The VLM describes visible scene conditions and uses "
                "Faster R-CNN detections only as unverified candidate context."
                "</div>"
            ),
            vlm_enabled_checkbox,
            vlm_tokens_widget,
            usar_prompt_box,
        ]
    )

    buttons = widgets.HBox(
        [
            mount_button,
            load_button,
            load_vlm_button,
            run_button,
            clear_button,
        ]
    )

    interface = widgets.VBox(
        [
            title,
            path_panel,
            input_panel,
            threshold_panel,
            vlm_panel,
            buttons,
            status,
            output,
        ],
        layout=widgets.Layout(
            width="100%"
        ),
    )

    display(interface)

    return interface


