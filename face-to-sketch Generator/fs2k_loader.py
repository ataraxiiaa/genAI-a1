"""
Task 4: FS2K Dataset Loader and Paired Data Augmentation Pipeline
Dataset:
- FS2K Facial Sketch Synthesis Dataset (2,104 paired photo-sketch images across 3 sketch styles)
- Official Splits: 1,058 train pairs, 1,046 test pairs
- Assignment Specification:
  - 15% of official train portion reserved as validation set using random seed 42
  - Validation split stratified by sketch style (Style 0, Style 1, Style 2)
  - Official test set strictly held out for final testing
  - Photographs and sketches resized to 128 x 128 pixels
  - Spatial augmentations (flipping, rotation) applied IDENTICALLY to both photo and sketch
"""

import os
import sys
import json
import random
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
NUM_STYLES = len(STYLE_NAMES)


def resolve_image_path(base_dir: str, subfolder: str, relative_path: str) -> Optional[str]:
    """
    Locates image on disk trying .jpg and .png extensions.
    Example:
        subfolder='photo', relative_path='photo1/image0110'
        -> e:/genAI-a1/FS2K/photo/photo1/image0110.jpg
    """
    for ext in ["", ".jpg", ".png", ".jpeg"]:
        full_path = os.path.join(base_dir, "FS2K", subfolder, relative_path + ext)
        if os.path.exists(full_path):
            return full_path
    return None


def get_sketch_relative_path(photo_relative_path: str) -> str:
    """
    Converts photo path to corresponding sketch path:
        'photo1/image0110' -> 'sketch1/sketch0110'
    """
    parts = photo_relative_path.split("/")
    folder = parts[0].replace("photo", "sketch")
    fname = parts[1].replace("image", "sketch")
    return f"{folder}/{fname}"


def create_stratified_train_val_split(
    anno_train_path: str,
    val_ratio: float = 0.15,
    seed: int = 42,
    manifest_cache_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates a deterministic 15% validation split stratified by sketch style (seed 42).
    Caches the resulting split to manifest_cache_path.
    """
    if manifest_cache_path and os.path.exists(manifest_cache_path):
        with open(manifest_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            return cached["train"], cached["val"]

    with open(anno_train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Group entries by style category
    by_style: Dict[int, List[Dict[str, Any]]] = {}
    for item in train_data:
        s = item["style"]
        by_style.setdefault(s, []).append(item)

    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []

    for s in sorted(by_style.keys()):
        items = list(by_style[s])
        rng = random.Random(seed)
        rng.shuffle(items)

        n_val = int(round(len(items) * val_ratio))
        val_items = items[:n_val]
        train_items = items[n_val:]

        val_split.extend(val_items)
        train_split.extend(train_items)

    if manifest_cache_path:
        os.makedirs(os.path.dirname(manifest_cache_path), exist_ok=True)
        with open(manifest_cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "val_ratio": val_ratio,
                "seed": seed,
                "total_train_available": len(train_data),
                "num_train": len(train_split),
                "num_val": len(val_split),
                "train": train_split,
                "val": val_split,
            }, f, indent=2)
        print(f"[FS2KLoader] Cached stratified validation manifest to: {manifest_cache_path}")

    return train_split, val_split


class FS2KDataset(Dataset):
    """
    PyTorch Dataset for paired Facial Photographs and Sketches from FS2K.
    Applies identical spatial transformations to both photograph and sketch to preserve
    pixel-to-pixel correspondence.
    """

    def __init__(
        self,
        base_dir: str,
        annotations: List[Dict[str, Any]],
        image_size: int = 128,
        mode: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.annotations = annotations
        self.image_size = (image_size, image_size)
        self.mode = mode
        self.seed = seed

    def __len__(self) -> int:
        return len(self.annotations)

    def _apply_identical_paired_augmentation(
        self,
        photo_pil: Image.Image,
        sketch_pil: Image.Image,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies IDENTICAL spatial transformations (horizontal flip, subtle rotation)
        to both photograph and sketch.
        """
        # 1. Resize to target dimensions
        photo = photo_pil.resize(self.image_size, Image.Resampling.BILINEAR)
        sketch = sketch_pil.resize(self.image_size, Image.Resampling.BILINEAR)

        # 2. Data Augmentation (Train mode only)
        if self.mode == "train":
            # Identical horizontal flip (p=0.5)
            if rng.random() > 0.5:
                photo = TF.hflip(photo)
                sketch = TF.hflip(sketch)

            # Identical subtle rotation (-8 to +8 degrees, p=0.4)
            if rng.random() > 0.6:
                angle = rng.uniform(-8.0, 8.0)
                photo = TF.rotate(photo, angle, interpolation=TF.InterpolationMode.BILINEAR)
                sketch = TF.rotate(sketch, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # 3. Convert to Tensor in range [0, 1]
        photo_t = TF.to_tensor(photo)
        sketch_t = TF.to_tensor(sketch)

        # 4. Standard GAN Normalization to [-1.0, 1.0]
        # Formula: (x - 0.5) / 0.5
        photo_norm = (photo_t - 0.5) / 0.5
        sketch_norm = (sketch_t - 0.5) / 0.5

        return photo_norm, sketch_norm

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.annotations[idx]
        photo_rel = item["image_name"]
        sketch_rel = get_sketch_relative_path(photo_rel)

        photo_path = resolve_image_path(self.base_dir, "photo", photo_rel)
        sketch_path = resolve_image_path(self.base_dir, "sketch", sketch_rel)

        if photo_path is None:
            raise FileNotFoundError(f"Missing photo image for {photo_rel} in {self.base_dir}")
        if sketch_path is None:
            raise FileNotFoundError(f"Missing sketch image for {sketch_rel} in {self.base_dir}")

        photo_pil = Image.open(photo_path).convert("RGB")
        sketch_pil = Image.open(sketch_path).convert("RGB")

        # Deterministic RNG for validation/test reproducibility; runtime random for training
        if self.mode == "train":
            rng = random.Random()
        else:
            rng = random.Random(self.seed + idx * 1337)

        photo_tensor, sketch_tensor = self._apply_identical_paired_augmentation(photo_pil, sketch_pil, rng)
        style_idx = int(item["style"])

        meta = {
            "image_name": photo_rel,
            "style": style_idx,
            "style_name": STYLE_NAMES[style_idx],
            "gender": item.get("gender", 0),
            "hair": item.get("hair", 0),
        }

        return photo_tensor, sketch_tensor, torch.tensor(style_idx, dtype=torch.long), meta


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a GAN tensor in [-1.0, 1.0] back to [0.0, 1.0] for display and metrics.
    """
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def get_fs2k_dataloaders(
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 128,
    val_ratio: float = 0.15,
    seed: int = 42,
    manifest_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, val, and test DataLoaders for the FS2K Face-to-Sketch dataset.
    Returns:
        train_loader: 899 paired training samples (85% of official train)
        val_loader:   159 paired validation samples (15% of official train, stratified)
        test_loader:  1046 paired test samples (official held-out test set)
    """
    anno_train = os.path.join(base_dir, "FS2K", "anno_train.json")
    anno_test = os.path.join(base_dir, "FS2K", "anno_test.json")

    manifest_cache = None
    if manifest_dir:
        manifest_cache = os.path.join(manifest_dir, "fs2k_val_split.json")

    train_data, val_data = create_stratified_train_val_split(
        anno_train_path=anno_train,
        val_ratio=val_ratio,
        seed=seed,
        manifest_cache_path=manifest_cache,
    )

    with open(anno_test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_dataset = FS2KDataset(base_dir, train_data, image_size=image_size, mode="train", seed=seed)
    val_dataset = FS2KDataset(base_dir, val_data, image_size=image_size, mode="val", seed=seed)
    test_dataset = FS2KDataset(base_dir, test_data, image_size=image_size, mode="test", seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FS2K Dataset Loader and Paired Augmentation")
    print("=" * 70)

    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifests = os.path.join(workspace, "manifests")

    train_ld, val_ld, test_ld = get_fs2k_dataloaders(
        base_dir=workspace,
        batch_size=8,
        num_workers=0,
        image_size=128,
        val_ratio=0.15,
        seed=42,
        manifest_dir=manifests,
    )

    print(f"Train Dataset Pairs: {len(train_ld.dataset):,} across {len(train_ld)} batches")
    print(f"Val Dataset Pairs:   {len(val_ld.dataset):,} across {len(val_ld)} batches (15% Stratified)")
    print(f"Test Dataset Pairs:  {len(test_ld.dataset):,} across {len(test_ld)} batches (Held out)")

    # Fetch a sample batch
    for photos, sketches, styles, metas in train_ld:
        print("\nBatch Verification:")
        print(f"  Photo Tensor:       {tuple(photos.shape)} | Range: [{photos.min():.2f}, {photos.max():.2f}]")
        print(f"  Sketch Tensor:      {tuple(sketches.shape)} | Range: [{sketches.min():.2f}, {sketches.max():.2f}]")
        print(f"  Style Labels:       {styles.tolist()}")
        print(f"  Sample 0 Meta:      {metas['image_name'][0]} -> {metas['style_name'][0]}")

        denorm = denormalize_image(photos)
        print(f"  Denormalized Range: [{denorm.min():.2f}, {denorm.max():.2f}]")
        break

    print("\nFS2K Paired Loader successfully verified!")
    print("=" * 70)
