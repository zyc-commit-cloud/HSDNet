# HSDNet

**Hypergraph-enhanced Small Object Detection Network for UAV Aerial Imagery**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

This repository provides the official PyTorch implementation of **HSDNet**, a hypergraph-enhanced detection framework designed for small-object detection in UAV aerial imagery.

HSDNet is developed from [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO). It strengthens small-object representations before substantial backbone downsampling, propagates shallow high-resolution spatial details through the detection neck, and provides additional overlap-based supervision for low-overlap small-object samples.

## Overview

Objects captured from UAV platforms are often small, densely distributed, and mutually occluded. Their weak visual cues may be progressively attenuated during successive backbone downsampling. Meanwhile, conventional detection pyramids do not fully exploit shallow high-resolution information, and small bounding boxes are highly sensitive to pixel-level localization deviations.

HSDNet addresses these challenges through three complementary improvements built upon the cross-level and cross-position hypergraph representation capability of Hyper-YOLO.

<details>
<summary><strong>Abstract</strong></summary>

Targets in UAV aerial imagery are typically small, densely distributed, and frequently occluded, posing challenges to spatial position perception, detail preservation during successive downsampling, effective neck-level feature propagation and fusion, and accurate small-object bounding-box regression.

To address these issues, we propose HSDNet, a hypergraph-enhanced framework for UAV small-object detection. First, a Detail- and Position-Aware Feature Enhancement (DPFE) module is introduced into Stages 2 and 3 of the backbone to strengthen small-object representations before further downsampling. It models local details and contextual cues across multiple patch scales while explicitly encoding horizontal and vertical positional information. Second, a high-resolution multi-scale detection neck shifts the detection scales toward higher resolutions and fuses shallow fine-grained spatial information with cross-stage semantic context enhanced by hypergraph reasoning. Finally, an auxiliary bounding-box regression strategy based on Inner-CIoU constructs scaled predicted and ground-truth boxes for overlap computation, providing additional overlap-based supervision for low-overlap small-object samples.

Experiments on VisDrone2019 and HazyDet demonstrate that HSDNet improves UAV small-object detection performance while reducing the parameter count relative to the Hyper-YOLO baseline.

</details>

## Main Contributions

### 1. Detail- and Position-Aware Feature Enhancement

The **Detail- and Position-Aware Feature Enhancement (DPFE)** module is introduced into Stages 2 and 3 of the backbone before further downsampling.

DPFE combines:

- PPA blocks for modeling local details and contextual information across different patch scales;
- a feature-selection mechanism for emphasizing informative patch responses; and
- branch- and fusion-level Coordinate Attention for encoding horizontal and vertical positional cues.

This design provides more discriminative and spatially coherent small-object features for subsequent multi-stage feature aggregation and hypergraph modeling.

### 2. High-resolution multi-scale detection neck

The original lowest-resolution prediction branch is replaced with a newly introduced high-resolution prediction branch.

For an input resolution of `640 × 640`, HSDNet performs detection at:

- `160 × 160`;
- `80 × 80`; and
- `40 × 40`.

The redesigned neck combines shallow fine-grained spatial details with cross-level semantic information enhanced through multi-stage feature aggregation and hypergraph computation.

### 3. Inner-CIoU-based auxiliary regression

The auxiliary bounding-box regression strategy constructs scaled predicted and ground-truth boxes using a fixed scaling ratio of `1.10`.

The auxiliary overlap calculation extends the effective overlap range and provides additional overlap-based supervision for low-IoU small-object samples.

## Results

### Main comparison

| Dataset | Model | mAP50 | mAP50-95 | Parameters | GFLOPs |
|---|---|---:|---:|---:|---:|
| VisDrone2019 | Hyper-YOLO-m | 0.430 | 0.264 | 33.34 M | 103.1 |
| VisDrone2019 | **HSDNet** | **0.493** | **0.304** | **26.41 M** | 131.2 |
| HazyDet | Hyper-YOLO-m | 0.750 | 0.546 | 33.33 M | 103.1 |
| HazyDet | **HSDNet** | **0.777** | **0.567** | **26.41 M** | 131.2 |

Compared with the Hyper-YOLO baseline:

- on VisDrone2019, HSDNet improves mAP50 by **6.3 percentage points** and mAP50-95 by **4.0 percentage points**;
- on HazyDet, HSDNet improves mAP50 by **2.7 percentage points** and mAP50-95 by **2.1 percentage points**;
- AP for small objects on VisDrone2019 increases from `0.129` to `0.184`, an absolute gain of **5.5 percentage points**; and
- the parameter count decreases from `33.34 M` to `26.41 M`, a reduction of approximately **20.8%**.

The increase in computational cost is mainly caused by feature processing and prediction at higher spatial resolutions.

### Scale-specific results on VisDrone2019

| Model | AP_S | AP_M | AP_L | GFLOPs |
|---|---:|---:|---:|---:|
| Hyper-YOLO | 0.129 | 0.349 | 0.464 | 103.1 |
| **HSDNet** | **0.184** | **0.384** | **0.476** | 131.2 |

The largest improvement is obtained for small objects, which is consistent with the intended design objective of HSDNet.

## Ablation Study

The ablation experiments were conducted on VisDrone2019 using Hyper-YOLO as the baseline.

- `M1`: DPFE module;
- `M2`: high-resolution multi-scale detection neck;
- `M3`: Inner-CIoU loss.

| Configuration | M1 | M2 | M3 | Precision | Recall | mAP50 | mAP50-95 | Parameters |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| S0 | No | No | No | 0.536 | 0.422 | 0.430 | 0.264 | 33.34 M |
| S1 | Yes | No | No | 0.546 | 0.428 | 0.439 | 0.267 | 34.04 M |
| S2 | No | Yes | No | 0.576 | 0.463 | 0.485 | 0.298 | 25.71 M |
| S3 | No | No | Yes | 0.543 | 0.418 | 0.433 | 0.265 | 33.34 M |
| S4 | Yes | Yes | No | **0.586** | 0.464 | 0.492 | 0.302 | 26.41 M |
| **S5 (HSDNet)** | **Yes** | **Yes** | **Yes** | 0.579 | **0.469** | **0.493** | **0.304** | **26.41 M** |

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/zyc-commit-cloud/HSDNet.git
cd HSDNet

conda create -n hsdnet python=3.10 -y
conda activate hsdnet

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

If necessary, install a PyTorch build compatible with the CUDA version available on your system.

## Experimental Setup

The experiments reported in the paper used the following settings:

| Item | Setting |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| CPU | Intel Xeon Gold 6430 |
| Framework | PyTorch |
| CUDA | 11.8 |
| Python | 3.10 |
| Input resolution | 640 × 640 |
| Training epochs | 200 |
| Batch size | 4 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.937 |

## Dataset Preparation

The experiments use the following two UAV object-detection datasets:

- [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset);
- [HazyDet](https://github.com/GrokCV/HazyDet).

The original annotations should be converted to the Ultralytics YOLO detection format before training.

### VisDrone2019

VisDrone2019 contains 10,209 static images across 10 object categories. The dataset is divided into:

- 6,471 training images;
- 548 validation images; and
- 3,190 testing images.

An example directory structure is:

```text
VisDrone2019/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Create a dataset configuration file such as `VisDrone.yaml`:

```yaml
path: /absolute/path/to/VisDrone2019

train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

### HazyDet

HazyDet contains 11,600 UAV images collected from real-world hazy scenes and synthetically generated hazy conditions, with approximately 383,000 annotated instances.

Prepare HazyDet in the same YOLO detection format and create a corresponding dataset YAML file. Refer to the official HazyDet repository for dataset download instructions and its original directory organization.

## Training

The HSDNet model configuration is located at:

```text
ultralytics/cfg/models/hsdnet/HSDNet.yaml
```

Train HSDNet using the command-line interface:

```bash
yolo detect train model=ultralytics/cfg/models/hsdnet/HSDNet.yaml data=/path/to/VisDrone.yaml epochs=200 imgsz=640 batch=4 device=0 optimizer=SGD lr0=0.01 momentum=0.937
```

Equivalent Python usage:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/hsdnet/HSDNet.yaml")

model.train(
    data="/path/to/VisDrone.yaml",
    epochs=200,
    imgsz=640,
    batch=4,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
)
```

Replace `/path/to/VisDrone.yaml` with the actual path to the dataset configuration file.

## Evaluation

Evaluate a trained model on the validation or test split:

```bash
yolo detect val model=/path/to/best.pt data=/path/to/VisDrone.yaml imgsz=640 batch=4 device=0
```

Equivalent Python usage:

```python
from ultralytics import YOLO

model = YOLO("/path/to/best.pt")

metrics = model.val(
    data="/path/to/VisDrone.yaml",
    imgsz=640,
    batch=4,
    device=0,
)
```

## Inference

Run inference on an image, directory, or video:

```bash
yolo detect predict model=/path/to/best.pt source=/path/to/images_or_video imgsz=640 device=0 save=True
```

Equivalent Python usage:

```python
from ultralytics import YOLO

model = YOLO("/path/to/best.pt")

model.predict(
    source="/path/to/images_or_video",
    imgsz=640,
    device=0,
    save=True,
)
```

Pretrained weights are not currently included in this repository. Replace `/path/to/best.pt` with the actual path to the trained model weights.

## Project Structure

The principal files related to HSDNet are:

```text
HSDNet/
├── ultralytics/
│   ├── cfg/models/hsdnet/HSDNet.yaml
│   ├── nn/modules/block.py
│   ├── nn/tasks.py
│   └── utils/loss.py
├── requirements.txt
├── setup.py
└── README.md
```

Their roles are:

- `HSDNet.yaml`: defines the HSDNet backbone and high-resolution detection neck;
- `block.py`: contains the DPFE, PPA, Coordinate Attention, and hypergraph-related modules;
- `tasks.py`: registers the model components and constructs the network;
- `loss.py`: contains the Inner-CIoU-based bounding-box regression implementation.

## Citation

If this work is useful in your research, please cite it. The bibliographic information will be updated after publication.

```bibtex
@article{zhou2026hsdnet,
  title  = {HSDNet: Hypergraph-enhanced Small Object Detection Network for UAV Aerial Imagery},
  author = {Zhou, Yucheng and Yang, Sen and Li, Wenyu and Tong, Jigang and Wang, Zenghui},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgements

This implementation is developed from:

- [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO);
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

We thank the authors and maintainers for making their work publicly available.

## License

This repository is released under the [GNU Affero General Public License v3.0](LICENSE).

## Contact

For questions, please open an issue in this repository or contact the corresponding author at [tjgtjut@163.com](mailto:tjgtjut@163.com).
