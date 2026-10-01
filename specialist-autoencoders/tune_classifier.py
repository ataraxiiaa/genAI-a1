"""
Hyperparameter Optimization using Optuna for Task 2: 4-Class Corruption Classifier
Oxford-IIIT Pet Dataset (128x128 RGB)

Tunes:
1. Learning rate (lr)
2. Batch size (batch_size)
3. Convolutional channel configuration (base_channels)
4. Dropout rate (dropout)
5. Weight decay (weight_decay)

Maximizes Validation Macro-F1 / Overall Accuracy and exports:
- checkpoints/best_classifier_hyperparameters.json
- checkpoints/optuna_classifier_study.db
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "universal-restoration"))

from classifier import CorruptionClassifier, NUM_CLASSES
from loader import get_dataloaders
from train_classifier import train_one_epoch, evaluate


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function for CorruptionClassifier.
    Tunes lr, batch_size, base_channels, dropout, and weight_decay.
    """
    # 1. Search Space (as mandated by assignment requirements)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_dir = os.path.join(base_dir, "manifests")

    # 2. DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 3. Model, Criterion, Optimizer
    model = CorruptionClassifier(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
        dropout_rate=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Multi-epoch evaluation with Median Pruning
    best_macro_f1 = 0.0
    for epoch in range(1, args.trial_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_res = evaluate(model, val_loader, criterion, device)

        macro_f1 = val_res["macro_f1"]
        val_acc = val_res["val_acc"]
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

        # Report metric to Optuna pruner
        trial.report(macro_f1, epoch)

        # Check for early pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Store auxiliary metrics as trial user attributes
    trial.set_user_attr("val_acc", float(val_acc))
    trial.set_user_attr("macro_f1", float(macro_f1))
    trial.set_user_attr("macro_precision", float(val_res["macro_precision"]))
    trial.set_user_attr("macro_recall", float(val_res["macro_recall"]))

    return macro_f1


def run_classifier_tuning(args):
    print("=" * 75)
    print("STARTING OPTUNA STUDY: 4-CLASS CORRUPTION CLASSIFIER")
    print("Search Space:")
    print("  - Learning Rate:               [1e-4, 3e-3] (log scale)")
    print("  - Batch Size:                  [16, 32, 64]")
    print("  - Conv Channel Configuration:  [16, 32, 48, 64]")
    print("  - Dropout Rate:                [0.0, 0.3]")
    print("  - Weight Decay:                [1e-5, 1e-2] (log scale)")
    print(f"Running {args.n_trials} Trials ({args.trial_epochs} epochs each)")
    print("=" * 75)

    checkpoints_dir = os.path.join(PARENT_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    db_path = os.path.join(checkpoints_dir, "optuna_classifier_study.db")
    storage_name = f"sqlite:///{os.path.abspath(db_path)}"

    study = optuna.create_study(
        study_name="task2_classifier_optimization",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n" + "=" * 75)
    print("OPTUNA CLASSIFIER STUDY FINISHED!")
    print("=" * 75)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Total Trials:     {len(study.trials)}")
    print(f"Complete Trials:  {len(complete_trials)}")
    print(f"Pruned Trials:    {len(pruned_trials)}")

    best_trial = study.best_trial
    print(f"\nBest Trial #{best_trial.number}:")
    print(f"  Target Metric (Macro-F1): {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    for k, v in best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    best_config_path = os.path.join(checkpoints_dir, "best_classifier_hyperparameters.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": best_trial.number,
                "best_macro_f1": best_trial.value,
                "user_attrs": best_trial.user_attrs,
                "best_params": best_trial.params,
            },
            f,
            indent=2,
        )

    print(f"\nBest hyperparameters saved to: {best_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Corruption Classifier Hyperparameters")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--trial_epochs", type=int, default=4, help="Epochs per trial")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()
    run_classifier_tuning(args)
