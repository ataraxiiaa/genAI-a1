"""
Task 4: Conditional GAN Face-to-Sketch Training Pipeline (Pix2Pix)
Features:
- Paired FS2K Dataset: 899 train pairs, 159 validation pairs (stratified 15%), 1,046 held-out test pairs
- Identical spatial augmentations (flipping, rotation) to preserve pixel correspondence
- Alternating adversarial training:
    1. Discriminator update: D(x, y, s) -> Real (1), D(x, G(x, s).detach(), s) -> Fake (0)
    2. Generator update: D(x, G(x, s), s) -> Real (1) + lambda_L1 * ||y - G(x, s)||_1
- Fixed validation photo visual sample tracking across training epochs
- Multi-metric evaluation: L1, SSIM, PSNR overall and per-style category
- CSV and Excel tracker logging
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any

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
from models import UNetGenerator, PatchGANDiscriminator
from losses import GeneratorLoss, DiscriminatorLoss, compute_metrics
from excel_tracker import append_cgan_epoch, append_style_eval


def evaluate_generator(
    netG: UNetGenerator,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluates generator across the stratified validation set.
    Returns:
        overall_metrics: mean L1 error, mean SSIM, mean PSNR
        per_style_metrics: breakdown for Style 1, Style 2, Style 3
    """
    netG.eval()
    total_samples = 0
    accum_l1 = 0.0
    accum_ssim = 0.0
    accum_psnr = 0.0

    style_accum = {s_idx: {"l1": 0.0, "ssim": 0.0, "psnr": 0.0, "count": 0} for s_idx in range(3)}

    with torch.no_grad():
        for photos, sketches, styles, _ in val_loader:
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

    overall = {
        "val_l1": accum_l1 / max(total_samples, 1),
        "val_ssim": accum_ssim / max(total_samples, 1),
        "val_psnr": accum_psnr / max(total_samples, 1),
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

    return overall, per_style


def extract_fixed_visual_samples(
    val_loader: DataLoader,
    samples_per_style: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Selects a deterministic fixed set of validation photographs (2 per style = 6 total)
    to inspect sketch synthesis progression over training time.
    """
    selected_by_style: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, str]]] = {0: [], 1: [], 2: []}

    for photos, sketches, styles, metas in val_loader:
        for i in range(len(styles)):
            s = int(styles[i].item())
            if len(selected_by_style[s]) < samples_per_style:
                img_name = metas["image_name"][i] if isinstance(metas["image_name"], list) else str(metas["image_name"])
                selected_by_style[s].append((photos[i], sketches[i], img_name))

        if all(len(v) == samples_per_style for v in selected_by_style.values()):
            break

    fixed_photos = []
    fixed_sketches = []
    fixed_styles = []
    fixed_names = []

    for s in [0, 1, 2]:
        for p, sk, name in selected_by_style[s]:
            fixed_photos.append(p)
            fixed_sketches.append(sk)
            fixed_styles.append(s)
            fixed_names.append(name)

    return (
        torch.stack(fixed_photos, dim=0),
        torch.stack(fixed_sketches, dim=0),
        torch.tensor(fixed_styles, dtype=torch.long),
        fixed_names,
    )


def save_visual_samples(
    netG: UNetGenerator,
    fixed_photos: torch.Tensor,
    fixed_sketches: torch.Tensor,
    fixed_styles: torch.Tensor,
    device: torch.device,
    save_path: str,
    epoch_title: str = "",
):
    """
    Saves fixed-validation inspection grid matching the Task 2/3 4-row layout:
    Row 0: Clean Ground Truth Sketch
    Row 1: Facial Photograph Input
    Row 2: Generated Sketch G(x, s)
    Row 3: Error Heatmap (|GT - Generated|) with magma colormap
    """
    netG.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.no_grad():
        photos_dev = fixed_photos.to(device)
        styles_dev = fixed_styles.to(device)
        fake_sketches = netG(photos_dev, styles_dev)

        # Denormalize from [-1, 1] to [0, 1]
        photos_01 = denormalize_image(photos_dev).cpu().permute(0, 2, 3, 1).numpy()
        real_sketches_01 = denormalize_image(fixed_sketches).permute(0, 2, 3, 1).numpy()
        fake_sketches_01 = denormalize_image(fake_sketches).cpu().permute(0, 2, 3, 1).numpy()

        # Compute pixel absolute error map
        error_map = np.abs(real_sketches_01 - fake_sketches_01)

    n_cols = len(fixed_photos)
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.8, 10), facecolor="white")
    titles = ["Target Sketch (GT)", "Photo Input", "Generated Sketch", "Error Heatmap (|GT - Fake|)"]
    rows = [real_sketches_01, photos_01, fake_sketches_01, error_map]

    for row_idx in range(4):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            img = rows[row_idx][col_idx]
            if row_idx == 3:
                gray_err = img.mean(axis=-1)
                ax.imshow(gray_err, cmap="magma", vmin=0.0, vmax=0.5)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.axis("off")

            # Column header shows style category on first row
            if row_idx == 0:
                s_name = STYLE_NAMES[int(fixed_styles[col_idx].item())]
                ax.set_title(f"Col {col_idx + 1}: {s_name}", fontsize=10, pad=4, color="#1B365D", fontweight="bold")

            # Row title on leftmost column
            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, titles[row_idx],
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold",
                    va="center", ha="right", rotation=90
                )

    main_title = "Conditional GAN: Face-to-Sketch Synthesis & Error Breakdown"
    if epoch_title:
        main_title += f" [{epoch_title}]"
    plt.suptitle(main_title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.close("all")


def run_training_pipeline(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Dataloaders
    print(f"Loading FS2K dataset from: {WORKSPACE_ROOT}")
    train_loader, val_loader, _ = get_fs2k_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=0.15,
        seed=args.seed,
        manifest_dir=manifest_dir,
    )

    # Fixed validation samples for tracking generator development
    fixed_photos, fixed_sketches, fixed_styles, _ = extract_fixed_visual_samples(val_loader, samples_per_style=2)
    print(f"Cached {len(fixed_photos)} fixed validation samples across all 3 styles for visual tracking.")

    # 2. Instantiate Models
    print(f"Building UNetGenerator (base_channels={args.base_channels}, embed_dim={args.embed_dim}, dropout={args.dropout})")
    netG = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Building PatchGANDiscriminator (base_channels={args.base_channels}, embed_dim={args.embed_dim})")
    netD = PatchGANDiscriminator(
        in_channels=6,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
    ).to(device)

    # 3. Objectives & Optimizers
    g_criterion = GeneratorLoss(lambda_l1=args.lambda_l1).to(device)
    d_criterion = DiscriminatorLoss(label_smoothing=args.label_smoothing).to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    # Cosine Annealing Schedulers
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, eta_min=args.g_lr * 0.05)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, eta_min=args.d_lr * 0.05)

    csv_log_path = os.path.join(args.save_dir, "cgan_training_log.csv")
    with open(csv_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,g_lr,d_lr,d_total,d_real,d_fake,g_total,g_adv,g_l1,val_l1,val_ssim,val_psnr\n")

    best_g_ckpt = os.path.join(args.save_dir, "best_cgan_generator.pt")
    best_d_ckpt = os.path.join(args.save_dir, "best_cgan_discriminator.pt")
    best_val_ssim = 0.0

    print("\n" + "=" * 75)
    print(f"STARTING CONDITIONAL GAN TRAINING ({args.epochs} EPOCHS, BATCH SIZE {args.batch_size})")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()

        d_accum = {"d_total": 0.0, "d_real": 0.0, "d_fake": 0.0}
        g_accum = {"g_total": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        total_batches = 0

        curr_g_lr = optimizer_G.param_groups[0]["lr"]
        curr_d_lr = optimizer_D.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch:02d}/{args.epochs:02d}]")
        for photos, sketches, styles, _ in pbar:
            photos = photos.to(device)
            sketches = sketches.to(device)
            styles = styles.to(device)

            # -----------------------------------------------------------------
            # 1. TRAIN DISCRIMINATOR: D(x, y, s) -> Real, D(x, G(x, s), s) -> Fake
            # -----------------------------------------------------------------
            optimizer_D.zero_grad()
            with torch.no_grad():
                fake_sketches = netG(photos, styles)

            d_real_logits = netD(photos, sketches, styles)
            d_fake_logits = netD(photos, fake_sketches.detach(), styles)

            loss_D, d_breakdown = d_criterion(d_real_logits, d_fake_logits)
            loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------------
            # 2. TRAIN GENERATOR: D(x, G(x, s), s) -> Real + lambda_L1 * L1(y, G)
            # -----------------------------------------------------------------
            optimizer_G.zero_grad()
            # Regenerate fake_sketches with gradients attached
            fake_sketches = netG(photos, styles)
            d_fake_for_g = netD(photos, fake_sketches, styles)

            loss_G, g_breakdown = g_criterion(d_fake_for_g, fake_sketches, sketches)
            loss_G.backward()
            optimizer_G.step()

            total_batches += 1
            for k in d_accum:
                d_accum[k] += d_breakdown[k]
            for k in g_accum:
                g_accum[k] += g_breakdown[k]

            pbar.set_postfix(
                D=f"{d_breakdown['d_total']:.3f}",
                G=f"{g_breakdown['g_total']:.2f}",
                L1=f"{g_breakdown['g_l1']:.3f}",
            )

        scheduler_G.step()
        scheduler_D.step()

        # Compute epoch training averages
        d_avg = {k: v / total_batches for k, v in d_accum.items()}
        g_avg = {k: v / total_batches for k, v in g_accum.items()}

        # Evaluate on stratified validation set
        val_overall, val_per_style = evaluate_generator(netG, val_loader, device)

        is_best = val_overall["val_ssim"] > best_val_ssim
        if is_best:
            best_val_ssim = val_overall["val_ssim"]
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": netG.state_dict(),
                    "val_ssim": val_overall["val_ssim"],
                    "val_psnr": val_overall["val_psnr"],
                    "val_l1": val_overall["val_l1"],
                    "per_style": val_per_style,
                    "config": vars(args),
                },
                best_g_ckpt,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "discriminator_state_dict": netD.state_dict(),
                    "config": vars(args),
                },
                best_d_ckpt,
            )

        print(
            f"[Epoch {epoch:02d}] Train: D={d_avg['d_total']:.3f} (R={d_avg['d_real']:.2f}, F={d_avg['d_fake']:.2f}), "
            f"G={g_avg['g_total']:.2f} (Adv={g_avg['g_adv']:.2f}, L1={g_avg['g_l1']:.3f}) | "
            f"Val: SSIM={val_overall['val_ssim']:.4f}, PSNR={val_overall['val_psnr']:.2f} dB, L1={val_overall['val_l1']:.4f}"
            f"{' [NEW BEST]' if is_best else ''}"
        )

        # Log to CSV
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{curr_g_lr:.6f},{curr_d_lr:.6f},"
                f"{d_avg['d_total']:.4f},{d_avg['d_real']:.4f},{d_avg['d_fake']:.4f},"
                f"{g_avg['g_total']:.4f},{g_avg['g_adv']:.4f},{g_avg['g_l1']:.4f},"
                f"{val_overall['val_l1']:.4f},{val_overall['val_ssim']:.4f},{val_overall['val_psnr']:.2f}\n"
            )

        # Log to Excel
        excel_path = os.path.join(CURRENT_DIR, "task4_gan_tracker.xlsx")
        append_cgan_epoch(
            epoch=epoch,
            g_lr=curr_g_lr,
            d_lr=curr_d_lr,
            d_metrics=d_avg,
            g_metrics=g_avg,
            val_metrics=val_overall,
            is_best=is_best,
            notes=f"New Best SSIM: {val_overall['val_ssim']:.4f}" if is_best else "",
            excel_path=excel_path,
        )

        # Save visual comparison periodically
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            sample_path = os.path.join(args.samples_dir, f"cgan_epoch_{epoch:02d}.png")
            save_visual_samples(
                netG, fixed_photos, fixed_sketches, fixed_styles, device, sample_path,
                epoch_title=f"Epoch {epoch:02d}"
            )
            print(f"  --> Saved fixed-validation visual samples to: {sample_path}")
            # Also log per-style metrics to Excel
            append_style_eval(f"Epoch {epoch:02d}", val_per_style, excel_path=excel_path)

    print("\n" + "=" * 75)
    print("CONDITIONAL GAN TRAINING COMPLETE!")
    print(f"Best Generator Checkpoint: {best_g_ckpt} (Val SSIM: {best_val_ssim:.4f})")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Train Style-Conditioned Face-to-Sketch GAN (Task 4)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--g_lr", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Reconstruction L1 weight")
    parser.add_argument("--label_smoothing", type=float, default=0.95, help="Discriminator label smoothing for real pairs")

    parser.add_argument("--base_channels", type=int, default=64, help="Base convolutional channel count")
    parser.add_argument("--embed_dim", type=int, default=16, help="Style embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial resolution")

    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="face-to-sketch Generator/visual_samples", help="Directory to save sample images")
    parser.add_argument("--sample_interval", type=int, default=5, help="Epoch interval to save visual samples")
    parser.add_argument("--config", type=str, default=None, help="Path to best_cgan_hyperparameters.json from Optuna")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA acceleration")

    args = parser.parse_args()

    # Load from Optuna JSON config if provided
    if args.config and os.path.exists(args.config):
        print(f"Loading best hyperparameters from Optuna config: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("best_params", cfg)
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    print(f"  Applied {k}: {v}")

    run_training_pipeline(args)


if __name__ == "__main__":
    main()
