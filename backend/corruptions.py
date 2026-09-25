"""
Image corruption engine for FastAPI backend (Tasks 1-3).
Works directly on PIL Images and NumPy arrays.
"""

import random
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Dict, Any, List

TEST_SEVERITY_CONFIGS = {
    "salt_and_pepper": {
        "low": 0.03,
        "medium": 0.08,
        "high": 0.15,
    },
    "gaussian_blur": {
        "low": {"kernel_size": 3, "radius": 0.7},
        "medium": {"kernel_size": 5, "radius": 1.5},
        "high": {"kernel_size": 7, "radius": 2.5},
    },
    "rectangular_occlusion": {
        "low": {"num_masks": 1, "target_area_ratio": 0.10},
        "medium": {"num_masks": 2, "target_area_ratio": 0.20},
        "high": {"num_masks": 3, "target_area_ratio": 0.35},
    },
}


def apply_salt_and_pepper_np(img_arr: np.ndarray, prob: float) -> np.ndarray:
    """Applies salt-and-pepper noise to an RGB image array in [0, 255]."""
    corrupted = img_arr.copy()
    H, W, _ = corrupted.shape
    total_pixels = H * W
    num_corrupted = int(total_pixels * prob)

    if num_corrupted == 0:
        return corrupted

    # Pick random pixel indices
    indices = np.random.choice(total_pixels, num_corrupted, replace=False)
    # Pick black (0) or white (255) with 50% probability
    values = np.random.choice([0, 255], num_corrupted, p=[0.5, 0.5])

    rows = indices // W
    cols = indices % W

    for r, c, val in zip(rows, cols, values):
        corrupted[r, c, :] = val

    return corrupted


def apply_gaussian_blur_pil(pil_img: Image.Image, radius: float) -> Image.Image:
    """Applies Gaussian blur using PIL GaussianBlur filter."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_rectangular_occlusion_np(
    img_arr: np.ndarray, num_masks: int, total_area_ratio: float
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """Applies 1 to 3 black rectangular masks covering the specified total area ratio."""
    corrupted = img_arr.copy()
    H, W, _ = corrupted.shape
    target_area = (H * W) * total_area_ratio
    area_per_mask = target_area / num_masks
    mask_coords = []

    for _ in range(num_masks):
        aspect = random.uniform(0.5, 2.0)
        h = int(np.sqrt(area_per_mask / aspect))
        w = int(h * aspect)
        h = max(2, min(H - 2, h))
        w = max(2, min(W - 2, w))

        top = random.randint(0, max(0, H - h))
        left = random.randint(0, max(0, W - w))

        corrupted[top : top + h, left : left + w, :] = 0
        mask_coords.append({"top": top, "left": left, "height": h, "width": w})

    return corrupted, mask_coords


def corrupt_image(
    pil_img: Image.Image,
    corruption_type: str,
    severity: str = "medium",
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Main corruption dispatcher.
    Returns:
        (corrupted_pil_img, metadata_dict)
    """
    # Resize to 128x128 RGB
    pil_128 = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)

    if corruption_type == "none" or corruption_type == "clean":
        return pil_128, {"type": "clean", "severity": severity}

    elif corruption_type == "salt_and_pepper":
        prob = TEST_SEVERITY_CONFIGS["salt_and_pepper"].get(severity, 0.08)
        arr = np.array(pil_128)
        corrupted_arr = apply_salt_and_pepper_np(arr, prob)
        return Image.fromarray(corrupted_arr), {"type": "salt_and_pepper", "severity": severity, "prob": prob}

    elif corruption_type == "gaussian_blur":
        cfg = TEST_SEVERITY_CONFIGS["gaussian_blur"].get(severity, {"radius": 1.5})
        radius = cfg["radius"]
        blurred = apply_gaussian_blur_pil(pil_128, radius=radius)
        return blurred, {"type": "gaussian_blur", "severity": severity, "radius": radius}

    elif corruption_type == "rectangular_occlusion":
        cfg = TEST_SEVERITY_CONFIGS["rectangular_occlusion"].get(severity, {"num_masks": 2, "target_area_ratio": 0.20})
        arr = np.array(pil_128)
        corrupted_arr, masks = apply_rectangular_occlusion_np(arr, cfg["num_masks"], cfg["target_area_ratio"])
        return Image.fromarray(corrupted_arr), {
            "type": "rectangular_occlusion",
            "severity": severity,
            "masks": masks,
        }

    else:
        return pil_128, {"type": "unknown", "severity": severity}
