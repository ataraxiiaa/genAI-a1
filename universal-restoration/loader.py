"""
Oxford-IIIT Pet Dataset Loader and Corruption Pipeline
For Tasks 1, 2, and 3 of Generative AI Assignment #1.

Features:
- Official 80/20 train/val split of trainval.txt with seed 42
- Test split from test.txt (held out until final evaluation)
- RGB conversion and 128x128 resize
- Runtime dynamic corruption generation for training (equal 25% distribution):
    0: Clean
    1: Salt-and-Pepper noise (prob in [0.02, 0.15])
    2: Gaussian blur (kernel in {3, 5, 7}, sigma in [0.5, 2.5])
    3: Rectangular occlusion (1-3 black masks covering 10%-35% area)
- Deterministic manifest generation and loading for validation and test splits
- Fixed severity levels for test evaluation (low, medium, high)
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


# Corruption Class Indices & Names
LABEL_CLEAN = 0
LABEL_SALT_PEPPER = 1
LABEL_GAUSSIAN_BLUR = 2
LABEL_OCCLUSION = 3

CORRUPTION_NAMES = [
    "clean",
    "salt_and_pepper",
    "gaussian_blur",
    "rectangular_occlusion",
]

# Fixed Test Severities defined by the assignment
TEST_SEVERITY_CONFIGS = {
    "salt_and_pepper": {
        "low": 0.03,
        "medium": 0.08,
        "high": 0.15,
    },
    "gaussian_blur": {
        "low": {"kernel_size": 3, "sigma": 0.7},
        "medium": {"kernel_size": 5, "sigma": 1.5},
        "high": {"kernel_size": 7, "sigma": 2.5},
    },
    "rectangular_occlusion": {
        "low": {"num_masks": 1, "target_area_ratio": 0.10},
        "medium": {"num_masks": 2, "target_area_ratio": 0.20},
        "high": {"num_masks": 3, "target_area_ratio": 0.35},
    },
}


# ==============================================================================
# Corruption Transform Implementations (Working on PyTorch Tensors in [0, 1])
# ==============================================================================

def apply_salt_and_pepper(
    img_tensor: torch.Tensor,
    prob: float,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Applies salt-and-pepper noise to an image tensor of shape (C, H, W) in [0, 1].
    Replaces a fraction `prob` of pixels with black (0.0) or white (1.0) with equal probability.
    """
    corrupted = img_tensor.clone()
    _, H, W = corrupted.shape
    total_pixels = H * W

    num_corrupted = int(total_pixels * prob)
    if num_corrupted == 0:
        return corrupted, {"prob": prob, "num_pixels": 0}

    if rng is not None:
        all_indices = list(range(total_pixels))
        selected_indices = rng.sample(all_indices, num_corrupted)
        is_white = [rng.random() < 0.5 for _ in range(num_corrupted)]
    else:
        selected_indices = torch.randperm(total_pixels)[:num_corrupted].tolist()
        is_white = (torch.rand(num_corrupted) < 0.5).tolist()

    for idx, white in zip(selected_indices, is_white):
        r = idx // W
        c = idx % W
        val = 1.0 if white else 0.0
        corrupted[:, r, c] = val

    meta = {"prob": float(prob), "num_pixels": num_corrupted}
    return corrupted, meta


def apply_gaussian_blur(
    img_tensor: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Applies Gaussian blur to an image tensor of shape (C, H, W) in [0, 1].
    Kernel size must be odd (3, 5, or 7).
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    corrupted = TF.gaussian_blur(
        img_tensor,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )
    meta = {"kernel_size": int(kernel_size), "sigma": float(sigma)}
    return corrupted, meta


def apply_rectangular_occlusion(
    img_tensor: torch.Tensor,
    num_masks: int,
    total_area_ratio: float,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Inserts `num_masks` black rectangular masks jointly covering `total_area_ratio` of the image.
    Locations are randomly sampled within the image boundaries.
    """
    corrupted = img_tensor.clone()
    _, H, W = corrupted.shape
    total_image_area = H * W
    target_area = total_image_area * total_area_ratio

    # Distribute target area among masks
    mask_coords = []
    area_per_mask = target_area / num_masks

    for _ in range(num_masks):
        # Choose aspect ratio for rectangle between 0.5 and 2.0
        aspect_ratio = (rng.uniform(0.5, 2.0) if rng else random.uniform(0.5, 2.0))
        h = int(np.sqrt(area_per_mask / aspect_ratio))
        w = int(h * aspect_ratio)

        # Clip dimensions to fit within image
        h = max(2, min(H - 2, h))
        w = max(2, min(W - 2, w))

        max_y = max(0, H - h)
        max_x = max(0, W - w)

        top = rng.randint(0, max_y) if rng else random.randint(0, max_y)
        left = rng.randint(0, max_x) if rng else random.randint(0, max_x)

        # Set mask region to 0.0 (black)
        corrupted[:, top : top + h, left : left + w] = 0.0
        mask_coords.append({"top": top, "left": left, "height": h, "width": w})

    meta = {
        "num_masks": num_masks,
        "total_area_ratio": float(total_area_ratio),
        "mask_coords": mask_coords,
    }
    return corrupted, meta


# ==============================================================================
# Dynamic Runtime Corruption Dispatcher
# ==============================================================================

def sample_runtime_corruption(
    clean_tensor: torch.Tensor,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Samples one of the 4 conditions with equal 25% probability:
    0: Clean
    1: Salt-and-Pepper (p uniform in [0.02, 0.15])
    2: Gaussian Blur (kernel in {3, 5, 7}, sigma uniform in [0.5, 2.5])
    3: Rectangular Occlusion (1-3 masks, total area uniform in [0.10, 0.35])
    """
    r = rng if rng is not None else random
    choice = r.choice([LABEL_CLEAN, LABEL_SALT_PEPPER, LABEL_GAUSSIAN_BLUR, LABEL_OCCLUSION])

    if choice == LABEL_CLEAN:
        return clean_tensor.clone(), LABEL_CLEAN, {"type": "clean"}

    elif choice == LABEL_SALT_PEPPER:
        prob = r.uniform(0.02, 0.15)
        corrupted, meta = apply_salt_and_pepper(clean_tensor, prob, rng=rng)
        meta["type"] = "salt_and_pepper"
        return corrupted, LABEL_SALT_PEPPER, meta

    elif choice == LABEL_GAUSSIAN_BLUR:
        kernel_size = r.choice([3, 5, 7])
        sigma = r.uniform(0.5, 2.5)
        corrupted, meta = apply_gaussian_blur(clean_tensor, kernel_size, sigma)
        meta["type"] = "gaussian_blur"
        return corrupted, LABEL_GAUSSIAN_BLUR, meta

    else:  # LABEL_OCCLUSION
        num_masks = r.randint(1, 3)
        area_ratio = r.uniform(0.10, 0.35)
        corrupted, meta = apply_rectangular_occlusion(clean_tensor, num_masks, area_ratio, rng=rng)
        meta["type"] = "rectangular_occlusion"
        return corrupted, LABEL_OCCLUSION, meta


# ==============================================================================
# Manifest Generation & Loading for Deterministic Validation/Testing
# ==============================================================================

def create_deterministic_record(
    image_name: str,
    corruption_type: str,
    severity_level: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Creates a deterministic parameter dictionary for an image."""
    rng = random.Random(seed)
    record: Dict[str, Any] = {
        "image_name": image_name,
        "corruption_type": corruption_type,
        "severity_level": severity_level,
        "seed": seed,
    }

    if corruption_type == "clean":
        record["label"] = LABEL_CLEAN

    elif corruption_type == "salt_and_pepper":
        record["label"] = LABEL_SALT_PEPPER
        if severity_level in TEST_SEVERITY_CONFIGS["salt_and_pepper"]:
            record["prob"] = TEST_SEVERITY_CONFIGS["salt_and_pepper"][severity_level]
        else:
            record["prob"] = rng.uniform(0.02, 0.15)

    elif corruption_type == "gaussian_blur":
        record["label"] = LABEL_GAUSSIAN_BLUR
        if severity_level in TEST_SEVERITY_CONFIGS["gaussian_blur"]:
            cfg = TEST_SEVERITY_CONFIGS["gaussian_blur"][severity_level]
            record["kernel_size"] = cfg["kernel_size"]
            record["sigma"] = cfg["sigma"]
        else:
            record["kernel_size"] = rng.choice([3, 5, 7])
            record["sigma"] = rng.uniform(0.5, 2.5)

    elif corruption_type == "rectangular_occlusion":
        record["label"] = LABEL_OCCLUSION
        if severity_level in TEST_SEVERITY_CONFIGS["rectangular_occlusion"]:
            cfg = TEST_SEVERITY_CONFIGS["rectangular_occlusion"][severity_level]
            num_masks = cfg["num_masks"]
            area_ratio = cfg["target_area_ratio"]
        else:
            num_masks = rng.randint(1, 3)
            area_ratio = rng.uniform(0.10, 0.35)

        record["num_masks"] = num_masks
        record["target_area_ratio"] = area_ratio

        # Generate deterministic mask coordinates for 128x128
        H, W = 128, 128
        area_per_mask = (H * W * area_ratio) / num_masks
        masks = []
        for _ in range(num_masks):
            aspect = rng.uniform(0.5, 2.0)
            h = int(np.sqrt(area_per_mask / aspect))
            w = int(h * aspect)
            h = max(2, min(H - 2, h))
            w = max(2, min(W - 2, w))
            top = rng.randint(0, max(0, H - h))
            left = rng.randint(0, max(0, W - w))
            masks.append({"top": top, "left": left, "height": h, "width": w})
        record["mask_coords"] = masks

    return record


def generate_validation_manifest(
    val_image_names: List[str],
    output_filepath: str,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a deterministic manifest for the validation set, ensuring equal
    distribution across the 4 conditions with fixed parameters per image.
    """
    rng = random.Random(seed)
    types = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
    manifest = []

    for i, name in enumerate(val_image_names):
        # Round-robin or uniform selection
        ctype = types[i % len(types)]
        img_seed = seed + i * 17
        rec = create_deterministic_record(name, ctype, seed=img_seed)
        manifest.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Validation manifest saved with {len(manifest)} entries to {output_filepath}")
    return manifest


def generate_test_manifests(
    test_image_names: List[str],
    manifest_dir: str,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generates fixed evaluation manifests for each corruption and severity level (low, medium, high).
    """
    os.makedirs(manifest_dir, exist_ok=True)
    paths = {}

    corruptions = ["salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
    severities = ["low", "medium", "high"]

    # 1. Clean test manifest
    clean_manifest = [create_deterministic_record(name, "clean", seed=seed + i) for i, name in enumerate(test_image_names)]
    clean_path = os.path.join(manifest_dir, "test_clean.json")
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=2)
    paths["clean"] = clean_path

    # 2. Corrupted manifests per type and severity
    for ctype in corruptions:
        for sev in severities:
            manifest = []
            for i, name in enumerate(test_image_names):
                rec = create_deterministic_record(name, ctype, severity_level=sev, seed=seed + i * 13)
                manifest.append(rec)
            key = f"{ctype}_{sev}"
            out_path = os.path.join(manifest_dir, f"test_{key}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            paths[key] = out_path

    print(f"Generated {len(paths)} test manifests in {manifest_dir}")
    return paths


def apply_record_to_tensor(clean_tensor: torch.Tensor, record: Dict[str, Any]) -> torch.Tensor:
    """Applies corruption described in a manifest record to a clean tensor."""
    ctype = record["corruption_type"]
    if ctype == "clean":
        return clean_tensor.clone()

    elif ctype == "salt_and_pepper":
        rng = random.Random(record["seed"])
        corrupted, _ = apply_salt_and_pepper(clean_tensor, record["prob"], rng=rng)
        return corrupted

    elif ctype == "gaussian_blur":
        corrupted, _ = apply_gaussian_blur(clean_tensor, record["kernel_size"], record["sigma"])
        return corrupted

    elif ctype == "rectangular_occlusion":
        corrupted = clean_tensor.clone()
        for mask in record.get("mask_coords", []):
            top, left = mask["top"], mask["left"]
            h, w = mask["height"], mask["width"]
            corrupted[:, top : top + h, left : left + w] = 0.0
        return corrupted

    return clean_tensor.clone()


# ==============================================================================
# Dataset Definition
# ==============================================================================

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset for Image Restoration (Tasks 1-3).
    Resizes images to 128x128 RGB.
    Modes:
      - 'train': runtime dynamic corruption (25% each of clean, noise, blur, occlusion)
      - 'val': deterministic corruption loaded from manifest or fixed seed per index
      - 'test': deterministic evaluation using test manifest
    """

    def __init__(
        self,
        image_dir: str,
        image_names: Optional[List[str]] = None,
        manifest_path: Optional[str] = None,
        mode: str = "train",
        target_size: Tuple[int, int] = (128, 128),
        seed: int = 42,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mode = mode
        self.target_size = target_size
        self.seed = seed
        self.manifest: Optional[List[Dict[str, Any]]] = None

        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.image_names = [item["image_name"] for item in self.manifest]
        elif image_names is not None:
            self.image_names = list(image_names)
        else:
            raise ValueError("Either image_names or a valid manifest_path must be provided.")

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_clean_image(self, name: str) -> torch.Tensor:
        # Check extensions (.jpg, .png, etc.)
        path = os.path.join(self.image_dir, name)
        if not os.path.exists(path):
            if os.path.exists(path + ".jpg"):
                path = path + ".jpg"
            elif os.path.exists(path + ".png"):
                path = path + ".png"

        img = Image.open(path).convert("RGB")
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        # Convert PIL to Tensor [0, 1] shape (3, H, W)
        tensor = TF.to_tensor(img)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            corrupted: Tensor (3, 128, 128)
            clean: Tensor (3, 128, 128)
            label: Tensor scalar long (0=clean, 1=salt_pepper, 2=blur, 3=occlusion)
            meta: dictionary with corruption info
        """
        name = self.image_names[idx]
        clean_tensor = self._load_clean_image(name)

        if self.mode == "train":
            # Dynamic runtime corruption
            corrupted, label, meta = sample_runtime_corruption(clean_tensor)
            meta["image_name"] = name
            return corrupted, clean_tensor, torch.tensor(label, dtype=torch.long), meta

        elif self.manifest is not None:
            # Deterministic from manifest
            record = self.manifest[idx]
            corrupted = apply_record_to_tensor(clean_tensor, record)
            label = record.get("label", LABEL_CLEAN)
            return corrupted, clean_tensor, torch.tensor(label, dtype=torch.long), record

        else:
            # Deterministic fallback without manifest using fixed seed per index
            item_seed = self.seed + idx * 31
            rng = random.Random(item_seed)
            corrupted, label, meta = sample_runtime_corruption(clean_tensor, rng=rng)
            meta["image_name"] = name
            return corrupted, clean_tensor, torch.tensor(label, dtype=torch.long), meta


# ==============================================================================
# Helper Functions to Prepare Splits and DataLoaders
# ==============================================================================

def load_split_image_names(annotations_dir: str) -> Tuple[List[str], List[str]]:
    """
    Loads image names from official Oxford-IIIT Pet annotation files:
    - trainval.txt: development data
    - test.txt: test data
    """
    trainval_file = os.path.join(annotations_dir, "trainval.txt")
    test_file = os.path.join(annotations_dir, "test.txt")

    trainval_names = []
    with open(trainval_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                trainval_names.append(parts[0])

    test_names = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                test_names.append(parts[0])

    return trainval_names, test_names


def get_dataset_splits(
    base_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Divides development collection (trainval.txt) into 80% train and 20% validation
    using random seed 42 as required by the assignment.
    Leaves official test.txt untouched.
    """
    annotations_dir = os.path.join(base_dir, "annotations")
    trainval_names, test_names = load_split_image_names(annotations_dir)

    rng = random.Random(seed)
    shuffled = list(trainval_names)
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_names = sorted(shuffled[:val_count])
    train_names = sorted(shuffled[val_count:])

    print(f"Dataset splits ready: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")
    return train_names, val_names, test_names


def pet_collate_fn(batch):
    """Custom collate function to handle variable metadata dictionaries gracefully."""
    corrupted = torch.stack([item[0] for item in batch], dim=0)
    clean = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.stack([item[2] if isinstance(item[2], torch.Tensor) else torch.tensor(item[2], dtype=torch.long) for item in batch], dim=0)
    meta = [item[3] for item in batch]
    return corrupted, clean, labels, meta




def get_dataloaders(
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    manifest_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Returns (train_loader, val_loader, test_names).
    Generates validation manifest if manifest_dir is provided and does not exist.
    """
    images_dir = os.path.join(base_dir, "images")
    if not os.path.exists(images_dir) and os.path.exists(os.path.join(base_dir, "Images")):
        images_dir = os.path.join(base_dir, "Images")

    train_names, val_names, test_names = get_dataset_splits(base_dir, val_ratio=0.2, seed=seed)

    # Setup validation manifest
    val_manifest_path = None
    if manifest_dir:
        val_manifest_path = os.path.join(manifest_dir, "val_manifest.json")
        if not os.path.exists(val_manifest_path):
            generate_validation_manifest(val_names, val_manifest_path, seed=seed)

    train_dataset = OxfordPetDataset(
        image_dir=images_dir,
        image_names=train_names,
        mode="train",
        seed=seed,
    )

    val_dataset = OxfordPetDataset(
        image_dir=images_dir,
        image_names=val_names,
        manifest_path=val_manifest_path,
        mode="val",
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pet_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pet_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_names


if __name__ == "__main__":
    # Quick sanity test
    base_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Testing loader from workspace: {base_workspace}")

    train_loader, val_loader, test_names = get_dataloaders(
        base_dir=base_workspace,
        batch_size=8,
        num_workers=0,
        manifest_dir=os.path.join(base_workspace, "manifests"),
        seed=42,
    )

    for corrupted, clean, labels, meta in train_loader:
        print(f"Batch corrupted shape: {corrupted.shape}")
        print(f"Batch clean shape: {clean.shape}")
        print(f"Batch labels: {labels.tolist()}")
        print(f"First image label name: {CORRUPTION_NAMES[labels[0].item()]}")
        break

    print("Loader sanity check successful!")
