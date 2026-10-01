"""
Training and Evaluation Pipeline for Task 2: 3 Specialist Autoencoders
Trains dedicated autoencoders exclusively on their single corruption domain:
1. NoiseSpecialistAE: Trained exclusively on Salt-and-Pepper noise
2. BlurSpecialistAE: Trained exclusively on Gaussian blur
3. OcclusionSpecialistAE: Trained exclusively on Rectangular box occlusions

Features:
- Dedicated Specialist Dataset with runtime domain-specific corruptions
- Deterministic validation tracking Val Loss, L1, SSIM, and PSNR (dB)
- Saves best checkpoint per specialist to checkpoints/best_specialist_{type}.pt
- Saves visual inspection grids (Clean vs Corrupted vs Restored vs Error Heatmap)
- Logs epoch progression to CSV for research paper reporting
"""

import os
import sys
import json
import random
import argparse

from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "universal-restoration"))

from specialists import (
    NoiseSpecialistAE,
    BlurSpecialistAE,
    OcclusionSpecialistAE,
    get_specialist,
    count_parameters,
)
from AE import UDAELoss
from loader import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    pet_collate_fn,
)
try:
    from track_specialists_excel import append_specialist_epoch
except ImportError:
    append_specialist_epoch = None



# ==============================================================================
# Domain-Specific Specialist Dataset
# ==============================================================================
class SpecialistPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset configured exclusively for a single corruption domain.
    Modes:
      - 'train': applies dynamic corruption within the specialist's target domain.
      - 'val': deterministic corruption for reproducible evaluation.
    """
    def __init__(
        self,
        image_dir: str,
        image_names: List[str],
        specialist_type: str,
        mode: str = "train",
        target_size: Tuple[int, int] = (128, 128),
        seed: int = 42,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.image_names = list(image_names)
        self.specialist_type = specialist_type.lower().strip()
        self.mode = mode
        self.target_size = target_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_clean(self, name: str) -> torch.Tensor:
        path = os.path.join(self.image_dir, name)
        if not os.path.exists(path):
            if os.path.exists(path + ".jpg"):
                path = path + ".jpg"
            elif os.path.exists(path + ".png"):
                path = path + ".png"
        img = Image.open(path).convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
        return TF.to_tensor(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        name = self.image_names[idx]
        clean_tensor = self._load_clean(name)

        if self.mode == "train":
            rng = random.Random()
        else:
            rng = random.Random(self.seed + idx * 37)

        if self.specialist_type in ["noise", "salt_and_pepper"]:
            prob = rng.uniform(0.02, 0.15)
            corrupted, meta = apply_salt_and_pepper(clean_tensor, prob, rng=rng)
            label = 1
            meta["type"] = "salt_and_pepper"

        elif self.specialist_type in ["blur", "gaussian_blur"]:
            kernel_size = rng.choice([3, 5, 7])
            sigma = rng.uniform(0.5, 2.5)
            corrupted, meta = apply_gaussian_blur(clean_tensor, kernel_size, sigma)
            label = 2
            meta["type"] = "gaussian_blur"

        elif self.specialist_type in ["occlusion", "rectangular_occlusion"]:
            num_masks = rng.randint(1, 3)
            area_ratio = rng.uniform(0.10, 0.35)
            corrupted, meta = apply_rectangular_occlusion(clean_tensor, num_masks, area_ratio, rng=rng)
            label = 3
            meta["type"] = "rectangular_occlusion"

        else:
            raise ValueError(f"Unknown specialist type: '{self.specialist_type}'")

        meta["image_name"] = name
        return corrupted, clean_tensor, torch.tensor(label, dtype=torch.long), meta



def get_specialist_dataloaders(
    base_dir: str,
    specialist_type: str,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Builds dedicated Train and Val loaders for a specific specialist."""
    img_dir = os.path.join(base_dir, "images")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(base_dir, "Images")

    # Load splits from trainval.txt
    split_file = os.path.join(base_dir, "annotations", "trainval.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]

    rng = random.Random(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    split_idx = int(0.80 * len(shuffled))
    train_names = shuffled[:split_idx]
    val_names = shuffled[split_idx:]

    train_ds = SpecialistPetDataset(img_dir, train_names, specialist_type, mode="train", seed=seed)
    val_ds = SpecialistPetDataset(img_dir, val_names, specialist_type, mode="val", seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pet_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pet_collate_fn,
    )
    return train_loader, val_loader


# ==============================================================================
# Training & Validation Loops
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: UDAELoss,
    device: torch.device,
    epoch: int,
    specialist_name: str,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"[{specialist_name}] Ep {epoch:02d} [Train]", leave=False)
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

        pbar.set_postfix({"Loss": f"{metrics['total_loss']:.4f}", "SSIM": f"{metrics['ssim']:.4f}"})

    return {
        "train_loss": running_loss / total_batches,
        "train_l1": running_l1 / total_batches,
        "train_ssim": running_ssim / total_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: UDAELoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    total_batches = len(dataloader)

    for corrupted, clean, labels, meta in dataloader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        restored = model(corrupted)
        loss, metrics = criterion(restored, clean)
        psnr_val = peak_signal_noise_ratio(restored, clean, data_range=1.0).item()

        running_loss += metrics["total_loss"]
        running_l1 += metrics["l1_loss"]
        running_ssim += metrics["ssim"]
        running_psnr += psnr_val

    return {
        "val_loss": running_loss / total_batches,
        "val_l1": running_l1 / total_batches,
        "val_ssim": running_ssim / total_batches,
        "val_psnr": running_psnr / total_batches,
    }


def save_visual_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 4,
):
    """Saves 4-row visual inspection grid: Clean vs Corrupted vs Restored vs Heatmap."""
    model.eval()
    batch = next(iter(dataloader))
    corrupted, clean, _, _ = batch
    corrupted = corrupted[:num_samples].to(device)
    clean = clean[:num_samples].to(device)

    with torch.no_grad():
        restored = model(corrupted)
        error_map = torch.abs(clean - restored)

    corrupted_np = corrupted.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    clean_np = clean.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    restored_np = restored.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    error_np = error_map.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()

    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 3, 10))
    titles = ["Clean Ground Truth", "Corrupted Input", "Specialist Restored", "Error Heatmap (|GT - Pred|)"]
    rows = [clean_np, corrupted_np, restored_np, error_np]

    for row_idx in range(4):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            img = rows[row_idx][col_idx]
            if row_idx == 3:
                gray_err = img.mean(axis=-1)
                ax.imshow(gray_err, cmap="magma", vmin=0.0, vmax=0.5)
            else:
                ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(titles[row_idx], fontsize=11, loc="left", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================================================================
# Train Single Specialist Routine
# ==============================================================================
def train_single_specialist(specialist_type: str, args):
    # Load shared Optuna architecture config if provided
    if hasattr(args, "config") and args.config and os.path.exists(args.config):
        print(f"Loading shared specialist architecture from Optuna config: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("best_params", cfg)
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    print(f"  Applied {k}: {v}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"STARTING TRAINING: {specialist_type.upper()} SPECIALIST AUTOENCODER")
    print("=" * 75)

    train_loader, val_loader = get_specialist_dataloaders(
        base_dir=base_dir,
        specialist_type=specialist_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = get_specialist(
        specialist_type,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
    ).to(device)
    params = count_parameters(model)
    print(f"Model: {model.name} | Parameters: {params:,} | Device: {device}")

    criterion = UDAELoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.save_dir, f"best_specialist_{specialist_type}.pt")
    log_csv_path = os.path.join(args.save_dir, f"training_log_{specialist_type}.csv")

    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,train_l1,train_ssim,val_loss,val_l1,val_ssim,val_psnr\n")

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, specialist_type)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_m["val_loss"])

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_m['train_loss']:.4f} (SSIM: {train_m['train_ssim']:.4f}) | "
            f"Val Loss: {val_m['val_loss']:.4f} | "
            f"Val SSIM: {val_m['val_ssim']:.4f} | "
            f"Val PSNR: {val_m['val_psnr']:.2f} dB | "
            f"LR: {curr_lr:.6f}"
        )

        with open(log_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{curr_lr:.6f},{train_m['train_loss']:.4f},{train_m['train_l1']:.4f},{train_m['train_ssim']:.4f},"
                f"{val_m['val_loss']:.4f},{val_m['val_l1']:.4f},{val_m['val_ssim']:.4f},{val_m['val_psnr']:.4f}\n"
            )

        is_best = val_m["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_m["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "specialist_type": specialist_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_m["val_loss"],
                    "val_ssim": val_m["val_ssim"],
                    "val_psnr": val_m["val_psnr"],
                    "config": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best {specialist_type} checkpoint! (Val SSIM: {val_m['val_ssim']:.4f})")

        # Log to Excel Tracker
        if append_specialist_epoch is not None:
            excel_path = os.path.join(CURRENT_DIR, "task2_specialists_tracker.xlsx")
            try:
                append_specialist_epoch(
                    specialist_type=specialist_type,
                    epoch=epoch,
                    lr=curr_lr,
                    train_metrics=train_m,
                    val_metrics=val_m,
                    is_best=is_best,
                    notes=f"New Best ({val_m['val_ssim']:.4f})" if is_best else "",
                    excel_path=excel_path,
                )
            except Exception as e:
                print(f"  Warning: Failed to log epoch to Excel: {e}")


        # Save visual sample periodically
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            sample_path = os.path.join(args.samples_dir, f"{specialist_type}_epoch_{epoch:02d}.png")
            save_visual_samples(model, val_loader, device, sample_path, num_samples=4)
            print(f"  --> Saved visual samples to: {sample_path}")

    print("\n" + "-" * 75)
    print(f"Completed {specialist_type.upper()} Specialist! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint: {best_ckpt_path}")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Train Specialist Autoencoders (Task 2)")
    parser.add_argument(
        "--specialist",
        type=str,
        default="noise",
        choices=["noise", "blur", "occlusion", "all"],
        help="Target specialist to train ('noise', 'blur', 'occlusion', or 'all')",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--alpha", type=float, default=0.75, help="Loss weight alpha (L1 vs SSIM)")
    parser.add_argument("--base_channels", type=int, default=64, help="Base convolutional channels")
    parser.add_argument("--latent_dim", type=int, default=256, help="Bottleneck latent dimension")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="visual_samples", help="Directory to save visual images")
    parser.add_argument("--sample_interval", type=int, default=5, help="Epoch interval to save visual samples")
    parser.add_argument("--config", type=str, default=None, help="Path to best_specialist_hyperparameters.json from Optuna")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")


    args = parser.parse_args()

    targets = ["noise", "blur", "occlusion"] if args.specialist == "all" else [args.specialist]
    for target in targets:
        train_single_specialist(target, args)


if __name__ == "__main__":
    main()
