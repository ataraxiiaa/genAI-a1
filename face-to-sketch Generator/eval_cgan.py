"""
Task 4: Evaluation and Visual Quality Assessment on Held-out FS2K Test Set (1,046 Pairs)
Features:
- Evaluates trained UNetGenerator on the strictly held-out test set (1,046 pairs)
- Computes L1 error, SSIM, and PSNR:
    - Overall mean
    - Per-style breakdown: Style 1 (Shaded Pencil), Style 2 (Clean Contour), Style 3 (Graphic Comic)
- Logs final test metrics to 'task4_gan_tracker.xlsx' under 'Style_Evaluation'
- Generates visual comparison grids displaying:
    [Target Ground Truth Sketch | Input Photograph | Generated Sketch | L1 Error Heatmap (Magma)]
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fs2k_loader import get_fs2k_dataloaders, denormalize_image, STYLE_NAMES
from models import UNetGenerator
from losses import compute_metrics
from excel_tracker import append_style_eval, EXCEL_PATH


def evaluate_test_set(
    netG: UNetGenerator,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[Dict]]:
    netG.eval()
    total_samples = 0
    accum_l1 = 0.0
    accum_ssim = 0.0
    accum_psnr = 0.0

    style_accum = {s_idx: {"l1": 0.0, "ssim": 0.0, "psnr": 0.0, "count": 0} for s_idx in range(3)}
    qualitative_samples = []

    print(f"\nEvaluating Generator on Held-Out Test Set ({len(test_loader.dataset)} pairs)...")
    with torch.no_grad():
        for b_idx, (photos, sketches, styles, metadata) in enumerate(tqdm(test_loader, desc="[Test Eval]")):
            b_size = photos.size(0)
            photos = photos.to(device)
            sketches = sketches.to(device)
            styles = styles.to(device)

            fake_sketches = netG(photos, styles)

            for i in range(b_size):
                l1_val, ssim_val, psnr_val = compute_metrics(
                    fake_sketches[i:i + 1], sketches[i:i + 1]
                )
                accum_l1 += l1_val
                accum_ssim += ssim_val
                accum_psnr += psnr_val
                total_samples += 1

                s = int(styles[i].item())
                style_accum[s]["l1"] += l1_val
                style_accum[s]["ssim"] += ssim_val
                style_accum[s]["psnr"] += psnr_val
                style_accum[s]["count"] += 1

                # Save 2 samples per style for high-res visual comparison
                if len([x for x in qualitative_samples if x["style"] == s]) < 2:
                    qualitative_samples.append({
                        "photo": photos[i].cpu(),
                        "sketch_gt": sketches[i].cpu(),
                        "sketch_fake": fake_sketches[i].cpu(),
                        "style": s,
                        "style_name": STYLE_NAMES[s],
                        "l1": l1_val,
                        "ssim": ssim_val,
                        "psnr": psnr_val,
                        "meta": metadata[i] if isinstance(metadata, list) else {},
                    })

    overall = {
        "test_l1": accum_l1 / max(total_samples, 1),
        "test_ssim": accum_ssim / max(total_samples, 1),
        "test_psnr": accum_psnr / max(total_samples, 1),
        "total_test_pairs": total_samples,
    }

    per_style = {}
    for s_idx in range(3):
        c = max(style_accum[s_idx]["count"], 1)
        per_style[STYLE_NAMES[s_idx]] = {
            "count": style_accum[s_idx]["count"],
            "l1": style_accum[s_idx]["l1"] / c,
            "ssim": style_accum[s_idx]["ssim"] / c,
            "psnr": style_accum[s_idx]["psnr"] / c,
        }

    return overall, per_style, qualitative_samples


def save_test_visual_grid(
    samples: List[Dict],
    save_path: str = "face-to-sketch Generator/visual_samples/test_qualitative_grid.png",
):
    """
    Saves qualitative test grid with 4 rows:
    Row 1: Ground Truth Sketch
    Row 2: Input Photo
    Row 3: Generated Sketch
    Row 4: L1 Error Heatmap (Magma)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_cols = len(samples)
    if num_cols == 0:
        return

    # Sort samples by style
    samples_sorted = sorted(samples, key=lambda x: x["style"])

    fig, axes = plt.subplots(4, num_cols, figsize=(num_cols * 3.0, 12.0), dpi=150)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, item in enumerate(samples_sorted):
        # Denormalize [-1, 1] -> [0, 1]
        photo_rgb = denormalize_image(item["photo"]).permute(1, 2, 0).numpy()
        gt_rgb = denormalize_image(item["sketch_gt"]).permute(1, 2, 0).numpy()
        fake_rgb = denormalize_image(item["sketch_fake"]).permute(1, 2, 0).numpy()

        # Compute error heatmap: mean across RGB channels
        error_map = np.mean(np.abs(fake_rgb - gt_rgb), axis=2)

        # Row 0: Ground Truth Sketch
        axes[0, col].imshow(gt_rgb)
        axes[0, col].set_title(f"{item['style_name']}\nGT Sketch", fontsize=9, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: Input Photograph
        axes[1, col].imshow(photo_rgb)
        axes[1, col].set_title("Input Photo", fontsize=9)
        axes[1, col].axis("off")

        # Row 2: Generated Sketch
        axes[2, col].imshow(fake_rgb)
        axes[2, col].set_title(
            f"Generated\nSSIM: {item['ssim']:.3f} | PSNR: {item['psnr']:.1f}dB",
            fontsize=8,
            color="navy",
            fontweight="semibold",
        )
        axes[2, col].axis("off")

        # Row 3: Error Heatmap (Magma colormap)
        im = axes[3, col].imshow(error_map, cmap="magma", vmin=0.0, vmax=0.5)
        axes[3, col].set_title(f"L1 Error: {item['l1']:.4f}", fontsize=8)
        axes[3, col].axis("off")

    # Add row labels on far left
    axes[0, 0].set_ylabel("Ground Truth", fontsize=10, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Input Photo", fontsize=10, fontweight="bold", labelpad=10)
    axes[2, 0].set_ylabel("Synthesized", fontsize=10, fontweight="bold", labelpad=10)
    axes[3, 0].set_ylabel("L1 Error Map", fontsize=10, fontweight="bold", labelpad=10)

    fig.suptitle(
        "FS2K Held-Out Test Set Qualitative Evaluation (Task 4: Conditional GAN)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    plt.close("all")
    print(f"Saved qualitative test grid to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate cGAN Generator on Held-Out FS2K Test Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_cgan_generator.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="checkpoints/best_cgan_hyperparameters.json", help="Path to hyperparams JSON")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial resolution")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--excel_path", type=str, default=EXCEL_PATH, help="Path to Excel tracker")
    parser.add_argument("--save_grid_path", type=str, default="face-to-sketch Generator/visual_samples/test_qualitative_grid.png")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA acceleration")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    base_channels = 64
    embed_dim = 16
    dropout = 0.0

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            p = cfg.get("best_params", cfg)
            base_channels = p.get("base_channels", base_channels)
            embed_dim = p.get("embed_dim", embed_dim)
            dropout = p.get("dropout", dropout)

    netG = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_styles=3,
        dropout=dropout,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("generator_state_dict", ckpt)
        netG.load_state_dict(state_dict)
        print(f"Loaded generator checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Evaluating initialized model.")

    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    _, _, test_loader = get_fs2k_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=0.15,
        seed=42,
        manifest_dir=manifest_dir,
    )

    overall, per_style, samples = evaluate_test_set(netG, test_loader, device)

    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET EVALUATION RESULTS (1,046 PAIRS)")
    print("=" * 75)
    print(f"Overall Mean SSIM: {overall['test_ssim']:.4f}")
    print(f"Overall Mean PSNR: {overall['test_psnr']:.2f} dB")
    print(f"Overall Mean L1:   {overall['test_l1']:.4f}")
    print("-" * 75)
    print("Per-Style Breakdown:")
    for s_name, res in per_style.items():
        print(f"  {s_name:25s} | SSIM: {res['ssim']:.4f} | PSNR: {res['psnr']:.2f} dB | L1: {res['l1']:.4f} ({res['count']} pairs)")
    print("=" * 75)

    # Log to Excel tracker
    append_style_eval("Held-out Test Set (Official 1,046 Pairs)", per_style, excel_path=args.excel_path)
    print(f"Appended test set metrics to Excel tracker: {args.excel_path}")

    # Save visual comparison grid
    save_test_visual_grid(samples, save_path=args.save_grid_path)


if __name__ == "__main__":
    main()
