# ROA-IQA

**ROA (Representative Observation Acquisition)** is a post-hoc calibration method that selects a fixed observation budget for multi-observation image quality assessment without retraining the IQA model. This repository demonstrates ROA using **YOTO-NR**.

The example uses a **self-trained YOTO-NR KonIQ-10k S01 checkpoint** (epoch 100; initialized from official Swin-T 22k weights). It compares **Native20** (20 crops) with **ROA-B7** (7 Sobol crops).

## 1. Setup

Use Python 3.11 and a CUDA-capable NVIDIA GPU. The reference predictions were generated on an RTX 4090 D with PyTorch 2.7.1 + CUDA 12.8.

```bash
python -m pip install -r requirements-inference.txt
```

Download the S01 checkpoint from this repository's **Releases** page after the release is published, then save it as `checkpoints/S01.pt`.

```bash
python scripts/verify_assets.py checkpoints/S01.pt
```

The model loader also requires the original [YOTO source](https://github.com/BarCodeReader/YOTO) and the [official Swin-T 22k initializer](https://github.com/SwinTransformer/storage/releases/tag/v1.0.8). Obtain these files from their original sources and prepare the local assets:

```bash
python scripts/prepare_yoto_assets.py --source-dir downloads/YOTO --swin downloads/swin_tiny_patch4_window7_224_22k.pth --asset-root assets
```

The repository does not redistribute the original YOTO source files.

## 2. Run and verify the inference example

```bash
python demo_yoto.py --asset-root assets --checkpoint checkpoints/S01.pt --image examples/inference/procedural_demo.png --output result.json
python verify_demo.py --actual result.json --golden examples/expected_outputs/golden.json
```

Reference predictions for the included procedural test image:

| Method | Predicted quality score |
|---|---:|
| Native20 | 0.506126880645752 |
| ROA-B7 | 0.5071967840194702 |

The verification script compares the image identity, observation coordinates, crop scores, and final predictions with the saved GPU reference (absolute tolerance `1e-5`). Small numerical differences across hardware or software environments may occur.

## 3. Optional: replay the calibration decision on CPU

```bash
python -m pip install -r requirements.txt
python replay_calibration.py --output replay_result.json
```

This replays existing KonIQ-10k S01 calibration predictions **without running the IQA model**. The reference selection uses 1,024 images; validation uses 4,096 images. The expected decision is `B_safe=7` and `Selected_B=7` (`alpha=0.05`).

**Scope:** The single-image example verifies inference outputs, not dataset-level PLCC/SRCC. No model training or new calibration is required for the example.

## Acknowledgment

[YOTO: You Only Train Once: A Unified Framework for Both Full-Reference and No-Reference Image Quality Assessment](https://github.com/BarCodeReader/YOTO). Please refer to the original repository and its authors for the YOTO model architecture.
