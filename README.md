# MorphoMoE

Official implementation of MorphoMoE.

## Overview

MorphoMoE is a morphology-oriented heterogeneous mixture of experts network for steel surface defect segmentation. It combines an LPDM-based shared encoder, three morphology-specialized experts, and a learned gating network to produce pixel-wise segmentation predictions.

## Architecture

MorphoMoE consists of four main components:

1. An LPDM-based shared encoder.
2. Morphology-oriented heterogeneous experts:
   - SDE for scale diversity;
   - RDE for directional representation;
   - CSE for channel-wise semantic enhancement.
3. A gating network that adaptively combines expert features.
4. A U-Net decoder that produces the segmentation logits.

## Environment

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The repository uses Python 3.8+ and the following core packages:

- PyTorch 2.0.0
- torchvision 0.15.1
- mamba-ssm 2.2.2

A CUDA-enabled PyTorch installation is recommended for training. Install the PyTorch and `mamba-ssm` builds that are compatible with your local CUDA environment.

## Dataset Preparation

## Dataset Preparation

Due to the size of the datasets and distribution considerations, this repository does not include the original dataset files.

The supported datasets can be obtained from the following public sources:

- **NEU-Seg**
  
  Dataset repository:
  https://github.com/DHW-Master/NEU_Seg

- **Magnetic Tile Dataset**
  
  Dataset repository:
  https://github.com/abin24/Magnetic-tile-defect-datasets.


After downloading the datasets, please organize the dataset paths according to your local environment and specify the corresponding image and mask directories through the YAML configuration files or command-line arguments.

The repository only provides dataset loading interfaces and does not include the original dataset files.

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

They provide placeholders for dataset directory paths and define training parameters such as epochs, batch size, learning rate, optimizer, scheduler, random seed, and output directory. The current MorphoMoE architecture settings use the defaults defined in `models/morphomoe.py`.

## Evaluation

Validation is performed during training. The reported segmentation metrics are:

- mIoU
- Dice
- Accuracy

## Results

Detailed experimental results are reported in the corresponding paper.
