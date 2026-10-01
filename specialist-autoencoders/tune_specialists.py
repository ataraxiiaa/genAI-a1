"""
Shared Hyperparameter Optimization using Optuna for Task 2: Specialist Autoencoders
Oxford-IIIT Pet Dataset (128x128 RGB)

Requirements:
- Performs a shared Optuna search to identify a common optimal architecture for the 3 specialists.
- Tunes:
    1. Learning rate (lr)
    2. Bottleneck size (latent_dim)
    3. Channel configuration (base_channels)
    4. Batch size (batch_size)
    5. L1-to-SSIM loss weighting (alpha)
- Exports winning configuration to checkpoints/best_specialist_hyperparameters.json
- The three specialists are then trained independently using this identified common architecture.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import optuna
from optuna.trial import TrialState

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "universal-restoration"))

from specialists import BaseSpecialistAE
from AE import UDAELoss
from loader import get_dataloaders
from train import train_one_epoch, validate_one_epoch


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function for identifying the shared specialist architecture.
    Tunes: lr, latent_dim, base_channels, batch_size, and loss alpha.
    """
    # 1. Search Space (as mandated by assignment requirements)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    alpha = trial.suggest_float("alpha", 0.60, 0.90, step=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_dir = os.path.join(base_dir, "manifests")

    # 2. DataLoaders with the sampled batch_size
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 3. Build Shared Architecture Candidate
    model = BaseSpecialistAE(
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout_rate=0.05,
    ).to(device)

    criterion = UDAELoss(alpha=alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Multi-epoch trial evaluation
    best_val_loss = float("inf")
    for epoch in range(1, args.trial_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics, breakdown = validate_one_epoch(model, val_loader, criterion, device, epoch)

        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("val_loss", float(best_val_loss))
    trial.set_user_attr("val_ssim", float(val_metrics["val_ssim"]))
    trial.set_user_attr("val_psnr", float(val_metrics["val_psnr"]))

    return best_val_loss


def run_specialist_tuning(args):
    print("=" * 75)
    print("STARTING SHARED OPTUNA STUDY: SPECIALIST ARCHITECTURE")
    print("Search Space:")
    print("  - Learning Rate:               [1e-4, 3e-3] (log scale)")
    print("  - Bottleneck Size:             [128, 256, 512]")
    print("  - Channel Configuration:       [32, 48, 64]")
    print("  - Batch Size:                  [16, 32, 64]")
    print("  - L1-to-SSIM Loss Alpha:       [0.60, 0.90]")
    print(f"Running {args.n_trials} Trials ({args.trial_epochs} epochs each)")
    print("=" * 75)

    checkpoints_dir = os.path.join(PARENT_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    db_path = os.path.join(checkpoints_dir, "optuna_specialist_study.db")
    storage_name = f"sqlite:///{os.path.abspath(db_path)}"

    study = optuna.create_study(
        study_name="task2_specialist_architecture_optimization",
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n" + "=" * 75)
    print("SHARED SPECIALIST OPTUNA STUDY FINISHED!")
    print("=" * 75)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Total Trials:     {len(study.trials)}")
    print(f"Complete Trials:  {len(complete_trials)}")
    print(f"Pruned Trials:    {len(pruned_trials)}")

    best_trial = study.best_trial
    print(f"\nOptimal Shared Specialist Architecture (Trial #{best_trial.number}):")
    print(f"  Best Val Loss: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    for k, v in best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    best_config_path = os.path.join(checkpoints_dir, "best_specialist_hyperparameters.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": best_trial.number,
                "best_val_loss": best_trial.value,
                "user_attrs": best_trial.user_attrs,
                "best_params": best_trial.params,
            },
            f,
            indent=2,
        )

    print(f"\nOptimal shared specialist hyperparameters saved to: {best_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared Optuna Search for Specialist Architecture")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--trial_epochs", type=int, default=4, help="Epochs per trial")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()
    run_specialist_tuning(args)
