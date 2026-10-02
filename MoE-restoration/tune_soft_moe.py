"""
Task 3: Optuna Hyperparameter Optimization for Soft MoE Restoration
Tunes:
1. Joint fine-tuning learning rate (joint_lr): [1e-5, 3e-4]
2. Softmax temperature tau: [0.5, 2.5]
3. Corruption classification loss weight (lambda_ce): [0.05, 0.4]
4. Routing balance regularizer weight (lambda_bal): [0.001, 0.05]
5. Reconstruction loss weighting (lambda_rec in [0.6, 0.9], lambda_ssim = 1.0 - lambda_rec)

Features:
- Optuna MedianPruner for early stopping poorly performing trials
- Routing collapse detector: prunes trials where routing collapses onto a single dominant branch
- Saves study results and best hyperparameters to JSON and Excel tracker
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import optuna
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path configuration
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECIALIST_DIR = os.path.join(WORKSPACE_ROOT, "specialist-autoencoders")
UNIVERSAL_DIR = os.path.join(WORKSPACE_ROOT, "universal-restoration")
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, SPECIALIST_DIR, UNIVERSAL_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from loader import get_dataloaders
from soft_moe import SoftMoERestoration
from losses import JointSoftMoELoss
from excel_tracker import append_optuna_trial


def is_routing_collapsed(weights_mean: torch.Tensor, max_threshold: float = 0.85, min_threshold: float = 0.02) -> bool:
    """
    Check if routing has collapsed:
    - Any single branch gets > 85% of weight on a balanced dataset
    - Any active branch gets starved to < 2%
    """
    max_w = torch.max(weights_mean).item()
    min_w = torch.min(weights_mean).item()
    return max_w > max_threshold or min_w < min_threshold


def evaluate_trial(
    model: SoftMoERestoration,
    val_loader: DataLoader,
    criterion: JointSoftMoELoss,
    device: torch.device,
    max_batches: int = 15,
) -> Dict[str, Any]:
    """Fast evaluation on a subset of validation batches during tuning."""
    model.eval()
    total_samples = 0
    accum_loss = 0.0
    accum_ssim = 0.0
    accum_mse = 0.0
    accum_weights = torch.zeros(4, device=device)

    with torch.no_grad():
        for b_idx, (corrupted, clean, labels, _) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            b = corrupted.size(0)
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            reconstruction, weights, logits = model(corrupted)
            loss, breakdown = criterion(reconstruction, clean, logits, labels, weights)

            total_samples += b
            accum_loss += breakdown["total_loss"] * b
            accum_ssim += breakdown["ssim"] * b
            accum_mse += torch.mean((reconstruction - clean) ** 2).item() * b
            accum_weights += weights.sum(dim=0)

    avg_mse = accum_mse / max(total_samples, 1)
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(avg_mse, 1e-10))).item()
    w_mean = (accum_weights / max(total_samples, 1)).cpu()

    return {
        "val_loss": accum_loss / max(total_samples, 1),
        "val_ssim": accum_ssim / max(total_samples, 1),
        "val_psnr": psnr,
        "weights_mean": w_mean,
        "collapsed": is_routing_collapsed(w_mean),
    }


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: Any,
) -> float:
    # 1. Sample Hyperparameters
    joint_lr = trial.suggest_float("joint_lr", 2e-5, 2e-4, log=True)
    tau = trial.suggest_float("temperature", 0.6, 2.0)
    lambda_ce = trial.suggest_float("lambda_ce", 0.05, 0.3)
    lambda_bal = trial.suggest_float("lambda_bal", 0.005, 0.05, log=True)
    lambda_rec = trial.suggest_float("lambda_rec", 0.70, 0.90)
    lambda_ssim = round(1.0 - lambda_rec, 3)

    # 2. Build Model & Load Checkpoints
    model = SoftMoERestoration(
        gate_base_channels=args.gate_base_channels,
        gate_dropout=args.gate_dropout,
        specialist_base_channels=args.specialist_base_channels,
        specialist_latent_dim=args.specialist_latent_dim,
        temperature=tau,
    ).to(device)

    cls_ckpt = os.path.join(args.checkpoint_dir, "best_corruption_classifier.pt")
    noise_ckpt = os.path.join(args.checkpoint_dir, "best_specialist_noise.pt")
    blur_ckpt = os.path.join(args.checkpoint_dir, "best_specialist_blur.pt")
    occ_ckpt = os.path.join(args.checkpoint_dir, "best_specialist_occlusion.pt")

    model.load_pretrained_weights(
        classifier_path=cls_ckpt,
        noise_path=noise_ckpt,
        blur_path=blur_ckpt,
        occlusion_path=occ_ckpt,
        device=device,
    )

    criterion = JointSoftMoELoss(
        lambda_rec=lambda_rec,
        lambda_ssim=lambda_ssim,
        lambda_ce=lambda_ce,
        lambda_bal=lambda_bal,
        num_branches=4,
    ).to(device)

    # 3. Quick Gate Warm-Up (1 epoch)
    model.freeze_experts()
    warmup_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    for b_idx, (corrupted, clean, labels, _) in enumerate(train_loader):
        if b_idx >= 30:  # Short warm-up slice for fast trial evaluation
            break
        corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
        warmup_opt.zero_grad()
        recon, w, logits = model(corrupted)
        loss, _ = criterion(recon, clean, logits, labels, w)
        loss.backward()
        warmup_opt.step()

    # 4. Joint Fine-Tuning Trials (e.g. 3 trial epochs)
    model.unfreeze_experts()
    joint_opt = torch.optim.AdamW(model.parameters(), lr=joint_lr, weight_decay=1e-4)

    best_ssim = 0.0
    final_eval = None

    for ep in range(1, args.trial_epochs + 1):
        model.train()
        for b_idx, (corrupted, clean, labels, _) in enumerate(train_loader):
            if b_idx >= 40:  # 40 batches per epoch is enough to detect signal/collapse
                break
            corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
            joint_opt.zero_grad()
            recon, w, logits = model(corrupted)
            loss, _ = criterion(recon, clean, logits, labels, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            joint_opt.step()

        # Intermediate evaluation
        final_eval = evaluate_trial(model, val_loader, criterion, device, max_batches=15)

        # Check for routing collapse
        if final_eval["collapsed"]:
            append_optuna_trial(
                trial_num=trial.number,
                params={
                    "lr": joint_lr, "tau": tau, "lambda_ce": lambda_ce,
                    "lambda_bal": lambda_bal, "lambda_rec": lambda_rec, "lambda_ssim": lambda_ssim,
                },
                val_ssim=final_eval["val_ssim"],
                val_psnr=final_eval["val_psnr"],
                val_loss=final_eval["val_loss"],
                routing_status="Collapsed",
                pruned=True,
            )
            raise optuna.exceptions.TrialPruned("Routing collapsed onto single branch!")

        # Optuna pruning check
        trial.report(final_eval["val_ssim"], step=ep)
        if trial.should_prune():
            append_optuna_trial(
                trial_num=trial.number,
                params={
                    "lr": joint_lr, "tau": tau, "lambda_ce": lambda_ce,
                    "lambda_bal": lambda_bal, "lambda_rec": lambda_rec, "lambda_ssim": lambda_ssim,
                },
                val_ssim=final_eval["val_ssim"],
                val_psnr=final_eval["val_psnr"],
                val_loss=final_eval["val_loss"],
                routing_status="Underperforming",
                pruned=True,
            )
            raise optuna.exceptions.TrialPruned()

        best_ssim = max(best_ssim, final_eval["val_ssim"])

    # Log successful trial to Excel
    append_optuna_trial(
        trial_num=trial.number,
        params={
            "lr": joint_lr, "tau": tau, "lambda_ce": lambda_ce,
            "lambda_bal": lambda_bal, "lambda_rec": lambda_rec, "lambda_ssim": lambda_ssim,
        },
        val_ssim=final_eval["val_ssim"],
        val_psnr=final_eval["val_psnr"],
        val_loss=final_eval["val_loss"],
        routing_status="Healthy",
        pruned=False,
    )

    return best_ssim


def main():
    parser = argparse.ArgumentParser(description="Optuna Tuning for Soft MoE Restoration")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--trial_epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_soft_moe_hyperparameters.json")
    parser.add_argument("--gate_base_channels", type=int, default=16)
    parser.add_argument("--gate_dropout", type=float, default=0.05)
    parser.add_argument("--specialist_base_channels", type=int, default=48)
    parser.add_argument("--specialist_latent_dim", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optuna Soft MoE study on device: {device}")

    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    storage_path = f"sqlite:///{os.path.join(args.checkpoint_dir, 'optuna_soft_moe_study.db')}"
    study = optuna.create_study(
        study_name="soft_moe_optimization",
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",  # Maximize Val SSIM
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, args),
        n_trials=args.n_trials,
        timeout=1800,  # 30 min max
    )

    print("\n" + "=" * 70)
    print("OPTUNA SOFT MOE STUDY COMPLETE!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Val SSIM: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:15s}: {v}")

    # Save to JSON
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": study.best_trial.number,
                "best_val_ssim": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    print(f"Saved best hyperparameters to: {args.save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
