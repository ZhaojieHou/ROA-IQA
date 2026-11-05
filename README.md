# ROA-IQA

**ROA (Representative Observation Acquisition)** is a post-hoc calibration method that selects a fixed observation budget for multi-observation image quality assessment without retraining the IQA model. This repository demonstrates ROA using **YOTO-NR**.

The example uses a **self-trained YOTO-NR KonIQ-10k S01 checkpoint** (epoch 100; initialized from official Swin-T 22k weights). It compares **Native20** (20 crops) with **ROA-B7** (7 Sobol crops).

## 1. Setup

Use Python 3.11 and a CUDA-capable NVIDIA GPU. The reference predictions were generated on an RTX 4090 D with PyTorch 2.7.1 + CUDA 12.8.

```bash
python -m pip install -r requirements-inference.txt
```

Download `YOTO_NR_KONIQ_S01_SELF_TRAINED.pt` from the [Releases page](https://github.com/ZhaojieHou/ROA-IQA/releases) after publication, then save it as `checkpoints/S01.pt`. The model is a separate Release attachment, not a Git file.

The model loader also requires the original [YOTO source](https://github.com/BarCodeReader/YOTO) and the [official Swin-T 22k initializer](https://github.com/SwinTransformer/storage/releases/tag/v1.0.8). Obtain these files from their original sources and prepare the local assets:

```bash
git clone https://github.com/BarCodeReader/YOTO.git downloads/YOTO
git -C downloads/YOTO checkout e9970581494cc426b99c6cf193127172c045ac1c
curl -L --fail https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth -o downloads/swin_tiny_patch4_window7_224_22k.pth
python scripts/prepare_yoto_assets.py --source-dir downloads/YOTO --swin downloads/swin_tiny_patch4_window7_224_22k.pth --asset-root assets
```

The repository does not redistribute the original YOTO source files. On Windows PowerShell, use `curl.exe` for the download command. The preparation script creates `assets/source/YOTO_official_snapshot/` and `assets/models/pretrained/`; keep the supplied vendored `timm` inside the YOTO source. Do not install the original training requirements. The loader retains tensor-shape checks and strict checkpoint loading.

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
