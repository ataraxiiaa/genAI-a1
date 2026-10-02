"""
Task 4: Optuna Hyperparameter Tuning for Face-to-Sketch Conditional GAN (Pix2Pix)
Explores the hyperparameter space to maximize validation SSIM:
- Generator learning rate (g_lr): 5e-5 to 4e-4
- Discriminator learning rate (d_lr): 5e-5 to 4e-4
- Batch size: [16, 32]
- Base channel count: [32, 64]
- Dropout rate: 0.0 to 0.2
- Style embedding dimension: [8, 16, 32]
- Reconstruction loss weight (lambda_l1): 50.0 to 150.0

Prunes underperforming trials with MedianPruner and records all trials in task4_gan_tracker.xlsx.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fs2k_loader import get_fs2k_dataloaders
from models import UNetGenerator, PatchGANDiscriminator, weights_init_normal
from losses import GeneratorLoss, DiscriminatorLoss
from train_cgan import evaluate_generator
from excel_tracker import append_optuna_trial, EXCEL_PATH


def objective(trial: optuna.Trial, device: torch.device, args: argparse.Namespace) -> float:
    # 1. Sample Hyperparameters
    g_lr = trial.suggest_float("g_lr", 5e-5, 4e-4, log=True)
    d_lr = trial.suggest_float("d_lr", 5e-5, 4e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32])
    lambda_l1 = trial.suggest_float("lambda_l1", 50.0, 150.0, step=10.0)

    # 2. Prepare DataLoaders
    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    train_loader, val_loader, _ = get_fs2k_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_ratio=0.15,
        seed=args.seed,
        manifest_dir=manifest_dir,
    )

    # 3. Instantiate Models & Apply Gaussian Initialization
    netG = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_styles=3,
        dropout=dropout,
    ).to(device)

    netD = PatchGANDiscriminator(
        in_channels=6,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_styles=3,
    ).to(device)

    weights_init_normal(netG)
    weights_init_normal(netD)

    # 4. Losses & Optimizers
    criterion_G = GeneratorLoss(lambda_l1=lambda_l1).to(device)
    criterion_D = DiscriminatorLoss(label_smoothing=0.95).to(device)

    opt_G = torch.optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))

    last_val_metrics = {"val_ssim": 0.0, "val_psnr": 0.0, "val_l1": 1.0}

    # 5. Training Loop for trial epochs
    for epoch in range(1, args.epochs_per_trial + 1):
        netG.train()
        netD.train()

        for photos, sketches, styles, _ in train_loader:
            photos = photos.to(device)
            sketches = sketches.to(device)
            styles = styles.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            d_real = netD(photos, sketches, styles)

            with torch.no_grad():
                fake_sketches = netG(photos, styles)

            d_fake = netD(photos, fake_sketches.detach(), styles)
            loss_d, _ = criterion_D(d_real, d_fake)
            loss_d.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_G.zero_grad()
            fake_sketches = netG(photos, styles)
            d_fake_for_g = netD(photos, fake_sketches, styles)
            loss_g, _ = criterion_G(d_fake_for_g, fake_sketches, sketches)
            loss_g.backward()
            opt_G.step()

        # Evaluate on validation split
        val_overall, _ = evaluate_generator(netG, val_loader, device)
        last_val_metrics = val_overall
        curr_ssim = val_overall["val_ssim"]

        # Report to Optuna for median pruning
        trial.report(curr_ssim, step=epoch)

        if trial.should_prune():
            # Log pruned trial to Excel
            try:
                append_optuna_trial(
                    trial_num=trial.number,
                    params=trial.params,
                    val_ssim=last_val_metrics["val_ssim"],
                    val_psnr=last_val_metrics["val_psnr"],
                    val_l1=last_val_metrics["val_l1"],
                    pruned=True,
                    excel_path=args.excel_path,
                )
            except Exception as e:
                print(f"[Optuna Excel Warning] Could not log pruned trial: {e}")
            raise optuna.TrialPruned()

    # Log completed trial to Excel
    try:
        append_optuna_trial(
            trial_num=trial.number,
            params=trial.params,
            val_ssim=last_val_metrics["val_ssim"],
            val_psnr=last_val_metrics["val_psnr"],
            val_l1=last_val_metrics["val_l1"],
            pruned=False,
            excel_path=args.excel_path,
        )
    except Exception as e:
        print(f"[Optuna Excel Warning] Could not log trial: {e}")

    return last_val_metrics["val_ssim"]


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Face-to-Sketch cGAN (Task 4)")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials to evaluate")
    parser.add_argument("--epochs_per_trial", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial image resolution")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 safe for Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_cgan_hyperparameters.json", help="Path to save best parameters JSON")
    parser.add_argument("--excel_path", type=str, default=EXCEL_PATH, help="Path to Excel tracker")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA acceleration")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("=" * 75)
    print("STARTING OPTUNA STUDY: CONDITIONAL GAN (TASK 4)")
    print(f"Device: {device} | Total Trials: {args.n_trials} | Epochs per Trial: {args.epochs_per_trial}")
    print("=" * 75)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(WORKSPACE_ROOT, 'checkpoints', 'optuna_cgan_study.db')}"

    study = optuna.create_study(
        study_name="cgan_face_to_sketch_study",
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",  # Maximize Val SSIM
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: objective(trial, device, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\n" + "=" * 75)
    print("OPTUNA STUDY COMPLETE!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Validation SSIM: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:18s}: {v}")
    print("=" * 75)

    # Save best parameters to JSON
    best_config = {
        "best_trial_number": study.best_trial.number,
        "best_val_ssim": study.best_value,
        "best_params": study.best_params,
    }
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best hyperparameters configuration to: {args.save_path}")

    # Also save copy in face-to-sketch Generator/ directory
    local_cfg_path = os.path.join(CURRENT_DIR, "best_cgan_hyperparameters.json")
    with open(local_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved local duplicate config to: {local_cfg_path}")


if __name__ == "__main__":
    main()
