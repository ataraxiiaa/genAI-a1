"""
Hyperparameter Optimization using Optuna for Task 1: Universal Denoising Autoencoder
Oxford-IIIT Pet Dataset (128x128 RGB)

Requirements from Assignment:
- Investigate at least:
  1. Learning rate (lr)
  2. Batch size (batch_size)
  3. Bottleneck dimension (latent_dim)
  4. Number of encoder channels (base_channels)
  5. Dropout rate (dropout)
  6. Loss-weight value alpha (alpha)
- Validation objective combining reconstruction quality and structural similarity
- Report complete search space, completed trials, best trial, and selected configuration
"""

import os
import sys
import json
import argparse
import optuna
from optuna.trial import TrialState
import torch

# Ensure current directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from AE import DenoisingAutoencoder, UDAELoss
from loader import get_dataloaders
from train import train_one_epoch, validate_one_epoch


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function:
    Samples hyperparameters, trains the model for a few epochs,
    and returns validation loss (combined L1 + SSIM) for Optuna to minimize.
    """
    # 1. Define Search Space as mandated by the assignment
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    alpha = trial.suggest_float("alpha", 0.6, 0.9, step=0.05)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_dir = os.path.join(base_dir, "manifests")

    # 2. Prepare DataLoaders with the sampled batch_size
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 3. Instantiate Model, Criterion, and Optimizer
    model = DenoisingAutoencoder(
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout_rate=dropout,
    ).to(device)

    criterion = UDAELoss(alpha=alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Train for a short number of trial epochs (e.g. 4-5 epochs per trial)
    best_val_loss = float("inf")

    for epoch in range(1, args.trial_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_metrics, _ = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # 5. Report intermediate metric for Optuna pruning
        trial.report(val_loss, step=epoch)

        # Handle pruning (stops poorly performing configurations early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def run_optuna_study(args):
    os.makedirs(args.save_dir, exist_ok=True)
    db_path = os.path.join(args.save_dir, "optuna_study.db")
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"

    print("=" * 70)
    print(f"Starting Optuna Study for Task 1: Universal Autoencoder")
    print(f"Number of Trials: {args.n_trials} | Epochs per Trial: {args.trial_epochs}")
    print(f"Search Space: lr, batch_size, base_channels, latent_dim, dropout, alpha")
    print("=" * 70)

    # MedianPruner kills trials that perform below the median of previous trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(
        study_name="universal_ae_optimization",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # Print and save summary
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 70)
    print("Optuna Study Results:")
    print(f"  Total Trials:    {len(study.trials)}")
    print(f"  Complete Trials: {len(complete_trials)}")
    print(f"  Pruned Trials:   {len(pruned_trials)}")
    print(f"\nBest Trial (Trial #{study.best_trial.number}):")
    print(f"  Best Validation Loss: {study.best_value:.4f}")
    print("  Optimal Hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"    {param_name:15s}: {param_value}")
    print("=" * 70)

    # Save best parameters to JSON for train.py to load directly
    best_params_file = os.path.join(args.save_dir, "best_hyperparameters.json")
    results_dict = {
        "best_trial_number": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params,
        "total_trials": len(study.trials),
        "complete_trials": len(complete_trials),
        "pruned_trials": len(pruned_trials),
    }

    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Best hyperparameters saved to: {best_params_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization (Task 1)")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials to run")
    parser.add_argument("--trial_epochs", type=int, default=3, help="Number of epochs per trial")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save Optuna db and json")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()
    run_optuna_study(args)
