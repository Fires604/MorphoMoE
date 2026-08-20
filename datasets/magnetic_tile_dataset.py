"""Portable reader for Magnetic Tile image and mask directories."""

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PairTransform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _paired_paths(image_dir: str, mask_dir: str) -> List[Tuple[Path, Path]]:
    """Pair image and mask files by filename stem in two caller-provided directories."""
    image_root, mask_root = Path(image_dir), Path(mask_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")

    images = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    masks = [
        path for path in mask_root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not images:
        raise RuntimeError(f"No image files found in: {image_root}")

    masks_by_stem = {}
    for path in masks:
        if path.stem in masks_by_stem:
            raise RuntimeError(f"Multiple masks share the same filename stem: {path.stem}")
        masks_by_stem[path.stem] = path

    missing = [path.stem for path in images if path.stem not in masks_by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing masks for image ids: {preview}")
    return [(path, masks_by_stem[path.stem]) for path in images]


class MagneticTileDataset(Dataset):
    """Read paired Magnetic Tile samples from user-supplied directories.

    ``transform``, when supplied, receives ``(image, mask)`` NumPy arrays and
    must return the transformed pair. The mask files must contain one integer
    class label per pixel.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: PairTransform = None,
    ) -> None:
        self.samples = _paired_paths(image_dir, mask_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim != 2:
            raise ValueError(f"Magnetic Tile masks must be single-channel labels: {mask_path}")

        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {"id": image_path.stem, "image": image_tensor, "mask": mask_tensor}


__all__ = ["MagneticTileDataset"]
