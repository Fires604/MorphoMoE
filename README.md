# MorphoMoE

Official implementation of **MorphoMoE: Resource-Efficient Morphology-Oriented Mixture of Experts for Robust Surface Defect Segmentation in Industrial IoT**.

This work is **currently under review**.

## Overview

MorphoMoE is a resource-efficient, morphology-oriented heterogeneous mixture-of-experts network for steel surface defect segmentation in Industrial IoT quality inspection. An LPDM-based shared encoder, three morphology-specialized experts, and a gating network produce pixel-wise defect maps under a compact parameter budget.

## Architecture

MorphoMoE consists of four main components:

1. A Local-Preserving Direction-Aware Mamba-CNN (LPDM) shared encoder.
2. Morphology-oriented heterogeneous experts:
   - **SDE** (Small Defect Expert) for local details and fine-grained structures;
   - **RDE** (Regional Defect Expert) for regional context and broad defect areas;
   - **CSE** (Complex Shape Expert) for irregular geometry and weak boundaries.
3. A gating network that fuses expert outputs with Softmax-normalized, input-dependent weights.
4. A shared U-Net decoder that produces the segmentation logits.

## Environment

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The repository uses Python 3.8+ and the following core packages:

- PyTorch 2.0.0
- torchvision 0.15.1
- mamba-ssm 2.2.2

A CUDA-enabled PyTorch installation is recommended for training. Install the PyTorch and `mamba-ssm` builds that are compatible with your local CUDA environment. The paper experiments used Python 3.8.10, CUDA 11.8, and a single NVIDIA RTX 4090.

## Dataset Preparation

This repository does not include the original dataset files. Obtain the public sources below, then set image and mask directories in the YAML configs or via command-line arguments.

- **NEU-Seg**: https://github.com/DHW-Master/NEU_Seg  
  4,470 images of size 200x200. The original split has 3,630 training images and 840 test images. Following the paper, further split the original training set into **2,904 training** and **726 validation** images. The test set (840 images) is held out for final reporting. Classes: Background, Inclusion, Patch, and Scratch (4 classes).

- **Magnetic Tile Dataset**: https://github.com/abin24/Magnetic-tile-defect-datasets.  
  1,344 images (392 defective, 952 normal). Resize all images to 256x256. Classes: Blowhole, Crack, Break, Fray, Uneven, and Free (6 classes).

**Augmentation.** In the paper, random augmentation is applied during training and disabled at test time. The released training script does not attach a default `transform`; pass one to the dataset class if you need to match the paper protocol.

**Early stopping.** Training runs for at most 200 epochs and stops if validation mIoU does not improve for **30** consecutive epochs (`early_stopping_patience: 30` in `configs/`).

## Training

Run training from the repository root with the unified entry point.

For NEU-Seg:

```bash
python -m train.train \
  --dataset neu \
  --train-image-dir /path/to/neu/train_images \
  --train-mask-dir /path/to/neu/train_masks \
  --val-image-dir /path/to/neu/val_images \
  --val-mask-dir /path/to/neu/val_masks
```

For Magnetic Tile Dataset:

```bash
python -m train.train \
  --dataset magnetic_tile \
  --train-image-dir /path/to/magnetic_tile/train_images \
  --train-mask-dir /path/to/magnetic_tile/train_masks \
  --val-image-dir /path/to/magnetic_tile/val_images \
  --val-mask-dir /path/to/magnetic_tile/val_masks
```

Use `--dry-run` to initialize the datasets, model, optimizer, and losses without starting training.

## Configuration

Configuration files are stored in `configs/`:

- `configs/neu.yaml`
- `configs/magnetic_tile.yaml`

They provide placeholders for dataset directory paths and define training parameters such as epochs, batch size, learning rate, optimizer, scheduler, early-stopping patience, random seed, and output directory. The current MorphoMoE architecture settings use the defaults defined in `models/morphomoe.py`.

## Evaluation

Validation is performed during training. The reported segmentation metrics are:

- mIoU
- Dice
- Accuracy

## Results

On the paper's test protocol, MorphoMoE uses **1.48 M** parameters and reports:

| Dataset | mIoU | mDice | Acc |
|---|---|---|---|
| NEU-Seg | 89.85% | 94.56% | 98.47% |
| Magnetic Tile | 79.18% | 87.95% | 99.42% |

Further comparisons, expert ablations, and robustness results under Gaussian blur and contrast reduction are given in the paper.
