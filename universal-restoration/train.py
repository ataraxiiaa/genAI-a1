"""
Training and Validation Pipeline for Task 1: Universal Denoising Autoencoder
Oxford-IIIT Pet Dataset (128x128 RGB)

Features:
- Full epoch training loop with dynamic runtime corruptions (25% each)
- Deterministic validation loop with metrics: Total Loss, L1, SSIM, PSNR
- Per-corruption validation breakdown (Clean, Noise, Blur, Occlusion)
- Saves best model checkpoint based on validation loss / SSIM
- Saves visual inspection grids: Clean vs Corrupted vs Restored vs Error Map
"""

import json
import os
import sys
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

# Ensure current directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from AE import DenoisingAutoencoder, UDAELoss
from loader import get_dataloaders, CORRUPTION_NAMES


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: UDAELoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Runs one full training epoch over the dataset."""
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for corrupted, clean, labels, meta in pbar:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        restored = model(corrupted)
        loss, metrics = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        running_loss += metrics["total_loss"]
        running_l1 += metrics["l1_loss"]
        running_ssim += metrics["ssim"]

        pbar.set_postfix({
            "Loss": f"{metrics['total_loss']:.4f}",
            "SSIM": f"{metrics['ssim']:.4f}",
        })

    avg_metrics = {
        "train_loss": running_loss / total_batches,
        "train_l1": running_l1 / total_batches,
        "train_ssim": running_ssim / total_batches,
    }
    return avg_metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: UDAELoss,
    device: torch.device,
    epoch: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluates the model on the deterministic validation dataset.
    Computes overall metrics and breaks down performance per corruption class.
    """
    model.eval()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    total_batches = len(dataloader)

    # Per-class accumulators
    class_stats = {name: {"count": 0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0} for name in CORRUPTION_NAMES}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
        for corrupted, clean, labels, meta in pbar:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            restored = model(corrupted)
            loss, metrics = criterion(restored, clean)
            psnr_val = peak_signal_noise_ratio(restored, clean, data_range=1.0).item()

            running_loss += metrics["total_loss"]
            running_l1 += metrics["l1_loss"]
            running_ssim += metrics["ssim"]
            running_psnr += psnr_val

            # Record per-sample corruption stats
            for i in range(len(labels)):
                lbl_name = CORRUPTION_NAMES[labels[i].item()]
                sample_restored = restored[i : i + 1]
                sample_clean = clean[i : i + 1]
                s_l1 = torch.nn.functional.l1_loss(sample_restored, sample_clean).item()
                s_ssim = structural_similarity_index_measure(sample_restored, sample_clean, data_range=1.0).item()
                s_psnr = peak_signal_noise_ratio(sample_restored, sample_clean, data_range=1.0).item()

                class_stats[lbl_name]["count"] += 1
                class_stats[lbl_name]["l1"] += s_l1
                class_stats[lbl_name]["ssim"] += s_ssim
                class_stats[lbl_name]["psnr"] += s_psnr

    overall_metrics = {
        "val_loss": running_loss / total_batches,
        "val_l1": running_l1 / total_batches,
        "val_ssim": running_ssim / total_batches,
        "val_psnr": running_psnr / total_batches,
    }

    # Normalize per-class stats
    breakdown = {}
    for name, stats in class_stats.items():
        cnt = max(1, stats["count"])
        breakdown[name] = {
            "l1": stats["l1"] / cnt,
            "ssim": stats["ssim"] / cnt,
            "psnr": stats["psnr"] / cnt,
            "count": stats["count"],
        }

    return overall_metrics, breakdown


def save_visual_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 4,
):
    """
    Saves visual comparison: Clean, Corrupted, Restored, and Absolute Error Map (|Restored - Clean|).
    """
    model.eval()
    with torch.no_grad():
        for corrupted, clean, labels, meta in dataloader:
            corrupted = corrupted.to(device)[:num_samples]
            clean = clean.to(device)[:num_samples]
            restored = model(corrupted)
            error_map = torch.abs(restored - clean)
            break

    # Stack vertically: Row 1=Clean, Row 2=Corrupted, Row 3=Restored, Row 4=Error Map
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2.5, 10))
    titles = ["Clean Ground-Truth", "Corrupted Input", "Restored Output", "Absolute Error Map"]

    for row_idx, tensor_batch in enumerate([clean, corrupted, restored, error_map]):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            img = tensor_batch[col_idx].cpu().permute(1, 2, 0).numpy()
            img = img.clip(0.0, 1.0)
            if row_idx == 3:
                # Show error map as colormap heatmap
                gray_err = img.mean(axis=-1)
                im = ax.imshow(gray_err, cmap="magma", vmin=0.0, vmax=0.5)
            else:
                ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(titles[row_idx], fontsize=11, loc="left", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(args):
    # If a config JSON is provided (or if checkpoints/best_hyperparameters.json exists), load it
    if hasattr(args, "config") and args.config and os.path.exists(args.config):
        print(f"Loading optimal hyperparameters from Optuna config: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("best_params", cfg)
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    print(f"  Applied {k}: {v}")

    # Set device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_dir = os.path.join(base_dir, "manifests")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading Oxford-IIIT Pet dataset from: {base_dir}")
    train_loader, val_loader, test_names = get_dataloaders(
        base_dir=base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 2. Build Model, Criterion, Optimizer
    print(f"Initializing DenoisingAutoencoder (base_channels={args.base_channels}, latent_dim={args.latent_dim}, dropout={args.dropout})")
    model = DenoisingAutoencoder(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = UDAELoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.save_dir, "best_universal_ae.pt")

    print("\n" + "=" * 70)
    print(f"Starting Training for {args.epochs} Epochs (Loss alpha={args.alpha})")
    print("=" * 70)

    # 3. Main Training & Validation Loop
    for epoch in range(1, args.epochs + 1):
        # A. Training Phase
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # B. Validation Phase
        val_metrics, breakdown = validate_one_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step(val_metrics["val_loss"])

        # C. Console Progress Report
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} (SSIM: {train_metrics['train_ssim']:.4f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val SSIM: {val_metrics['val_ssim']:.4f} | "
            f"Val PSNR: {val_metrics['val_psnr']:.2f} dB"
        )

        # Print per-corruption validation breakdown
        breakdown_str = " | ".join(
            [f"{name[:5]}: SSIM={stats['ssim']:.3f}, PSNR={stats['psnr']:.1f}" for name, stats in breakdown.items()]
        )
        print(f"           Breakdown -> {breakdown_str}")

        # D. Save Best Model Checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_ssim": val_metrics["val_ssim"],
                    "val_psnr": val_metrics["val_psnr"],
                    "config": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path}")

        # E. Save Visual Inspection Image Every few Epochs
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            sample_img_path = os.path.join(args.samples_dir, f"epoch_{epoch:02d}_restoration.png")
            save_visual_samples(model, val_loader, device, sample_img_path, num_samples=4)
            print(f"  --> Saved visual inspection samples to {sample_img_path}")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Universal Denoising Autoencoder (Task 1)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--alpha", type=float, default=0.8, help="Loss weight alpha (L1 vs SSIM)")
    parser.add_argument("--base_channels", type=int, default=32, help="Base convolutional channels")
    parser.add_argument("--latent_dim", type=int, default=256, help="Bottleneck latent dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="visual_samples", help="Directory to save visual images")
    parser.add_argument("--sample_interval", type=int, default=3, help="Epoch interval to save visual samples")
    parser.add_argument("--config", type=str, default=None, help="Path to best_hyperparameters.json from Optuna")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training")

    args = parser.parse_args()
    train(args)
