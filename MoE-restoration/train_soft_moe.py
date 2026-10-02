"""
Task 3: Joint Training Pipeline for Soft Mixture-of-Experts (MoE) Restoration
Training Workflow:
1. Model Initialization:
   - Gating network initialized from Task 2 pre-trained classifier
   - 3 Specialist Autoencoders initialized from Task 2 pre-trained checkpoints
2. Stage 1: Gate Warm-Up (Experts Frozen):
   - Experts parameters frozen (requires_grad = False)
   - Gate trained via reconstruction error + classification cross-entropy
3. Stage 2: End-to-End Joint Fine-Tuning:
   - Experts unfrozen (requires_grad = True)
   - Complete system trained jointly with a smaller fine-tuning learning rate
   - Multi-component joint loss: L1, SSIM, Cross-Entropy, and Routing Balance
4. Comprehensive logging to CSV, Excel Tracker, and visual sample generation
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure workspace and subpackage paths are configured
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECIALIST_DIR = os.path.join(WORKSPACE_ROOT, "specialist-autoencoders")
UNIVERSAL_DIR = os.path.join(WORKSPACE_ROOT, "universal-restoration")
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, SPECIALIST_DIR, UNIVERSAL_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from loader import get_dataloaders
from soft_moe import SoftMoERestoration, BRANCH_SHORT_NAMES
from losses import JointSoftMoELoss
from excel_tracker import append_warmup_epoch, append_joint_epoch


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-10:
        return 50.0
    return 10.0 * torch.log10(torch.tensor((max_val ** 2) / mse)).item()


def evaluate_soft_moe(
    model: SoftMoERestoration,
    val_loader: DataLoader,
    criterion: JointSoftMoELoss,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    Evaluate Soft MoE model across validation dataset.
    Returns:
        metrics_dict: Aggregated loss, L1, SSIM, PSNR, CE, Balance loss, and routing distribution
        avg_weights: (4,) tensor of average routing weights assigned across validation set
    """
    model.eval()
    total_samples = 0
    accum_loss = 0.0
    accum_l1 = 0.0
    accum_ssim = 0.0
    accum_mse = 0.0
    accum_ce = 0.0
    accum_bal = 0.0
    accum_weights = torch.zeros(4, device=device)
    correct_gate_preds = 0

    with torch.no_grad():
        for corrupted, clean, labels, _ in val_loader:
            b_size = corrupted.size(0)
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            reconstruction, weights, logits = model(corrupted)
            loss, breakdown = criterion(reconstruction, clean, logits, labels, weights)

            total_samples += b_size
            accum_loss += breakdown["total_loss"] * b_size
            accum_l1 += breakdown["loss_l1"] * b_size
            accum_ssim += breakdown["ssim"] * b_size
            accum_ce += breakdown["loss_ce"] * b_size
            accum_bal += breakdown["loss_bal"] * b_size
            accum_mse += torch.mean((reconstruction - clean) ** 2).item() * b_size
            accum_weights += weights.sum(dim=0)

            preds = torch.argmax(logits, dim=-1)
            correct_gate_preds += (preds == labels).sum().item()

    avg_mse = accum_mse / total_samples
    avg_psnr = compute_psnr(avg_mse)
    avg_weights = (accum_weights / total_samples).cpu()
    gate_acc = correct_gate_preds / total_samples

    metrics = {
        "total_loss": accum_loss / total_samples,
        "loss_l1": accum_l1 / total_samples,
        "ssim": accum_ssim / total_samples,
        "psnr": avg_psnr,
        "loss_ce": accum_ce / total_samples,
        "loss_bal": accum_bal / total_samples,
        "gate_acc": gate_acc,
        "weights_mean": avg_weights.tolist(),
    }
    return metrics, avg_weights


def save_visual_samples(
    model: SoftMoERestoration,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    epoch_title: str = "",
    num_samples: int = 4,
):
    """
    Save 4-row visual inspection grid matching reference design:
    Row 0: Clean Ground Truth
    Row 1: Corrupted Input
    Row 2: Soft MoE Restored
    Row 3: Error Heatmap (|GT - Pred|) with magma colormap
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    samples_by_class = {}
    with torch.no_grad():
        for corrupted, clean, labels, _ in val_loader:
            for i in range(len(labels)):
                c = int(labels[i].item())
                if c not in samples_by_class:
                    samples_by_class[c] = (corrupted[i:i + 1], clean[i:i + 1])
                if len(samples_by_class) == 4:
                    break
            if len(samples_by_class) == 4:
                break

    # If 4 classes found, use one of each (clean, noise, blur, occlusion); otherwise fallback to first batch
    if len(samples_by_class) == 4:
        corrupted_t = torch.cat([samples_by_class[c][0] for c in sorted(samples_by_class.keys())], dim=0).to(device)
        clean_t = torch.cat([samples_by_class[c][1] for c in sorted(samples_by_class.keys())], dim=0).to(device)
    else:
        for corrupted_b, clean_b, _, _ in val_loader:
            corrupted_t = corrupted_b[:num_samples].to(device)
            clean_t = clean_b[:num_samples].to(device)
            break

    with torch.no_grad():
        restored_t, _, _ = model(corrupted_t)
        error_map_t = torch.abs(clean_t - restored_t)

    clean_np = clean_t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    corrupted_np = corrupted_t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    restored_np = restored_t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    error_np = error_map_t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()

    n_cols = clean_np.shape[0]
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 3, 10), facecolor="white")
    titles = ["Clean Ground Truth", "Corrupted Input", "Soft MoE Restored", "Error Heatmap (|GT - Pred|)"]
    rows = [clean_np, corrupted_np, restored_np, error_np]

    for row_idx in range(4):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
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
    plt.savefig(save_path, dpi=150, facecolor="white", edgecolor="none")
    plt.close()


def run_training_pipeline(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading Oxford-IIIT Pet balanced dataset from: {WORKSPACE_ROOT}")
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # 2. Build Soft MoE Model
    print("Initializing SoftMoERestoration Architecture...")
    model = SoftMoERestoration(
        gate_base_channels=args.gate_base_channels,
        gate_dropout=args.gate_dropout,
        specialist_base_channels=args.specialist_base_channels,
        specialist_latent_dim=args.specialist_latent_dim,
        temperature=args.temperature,
    ).to(device)

    # 3. Load Pre-trained Weights from Task 2
    if not args.scratch:
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

    # 4. Joint Loss Function
    criterion = JointSoftMoELoss(
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_ce=args.lambda_ce,
        lambda_bal=args.lambda_bal,
        num_branches=4,
    ).to(device)

    csv_log_path = os.path.join(args.save_dir, "soft_moe_training_log.csv")
    with open(csv_log_path, "w", encoding="utf-8") as f:
        f.write("stage,epoch,lr,train_total_loss,train_l1,train_ssim,train_ce,train_bal,"
                "val_total_loss,val_l1,val_ssim,val_psnr,val_ce,val_bal,gate_acc,"
                "w_clean,w_noise,w_blur,w_occ\n")

    best_ckpt_path = os.path.join(args.save_dir, "best_soft_moe.pt")
    best_val_loss = float("inf")
    best_val_ssim = 0.0

    # =========================================================================
    # STAGE 1: GATE WARM-UP (EXPERTS FROZEN)
    # =========================================================================
    if args.warmup_epochs > 0:
        print("\n" + "=" * 75)
        print(f"STAGE 1: GATE WARM-UP ({args.warmup_epochs} EPOCHS, EXPERTS FROZEN)")
        print("=" * 75)
        model.freeze_experts()

        warmup_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.warmup_epochs + 1):
            model.train()
            train_loss_accum = 0.0
            total_train = 0

            pbar = tqdm(train_loader, desc=f"[Warm-Up] Epoch {epoch:02d}/{args.warmup_epochs:02d}")
            for corrupted, clean, labels, _ in pbar:
                corrupted = corrupted.to(device)
                clean = clean.to(device)
                labels = labels.to(device)

                warmup_optimizer.zero_grad()
                reconstruction, weights, logits = model(corrupted)
                loss, breakdown = criterion(reconstruction, clean, logits, labels, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                warmup_optimizer.step()

                b = corrupted.size(0)
                total_train += b
                train_loss_accum += loss.item() * b
                pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{breakdown['ssim']:.4f}")

            val_m, val_w = evaluate_soft_moe(model, val_loader, criterion, device)
            train_loss = train_loss_accum / total_train

            print(
                f"[Warm-Up Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_m['total_loss']:.4f} | Val SSIM: {val_m['ssim']:.4f} | "
                f"Gate Acc: {val_m['gate_acc'] * 100:.2f}% | "
                f"Weights: Clean={val_w[0]:.2f}, Noise={val_w[1]:.2f}, Blur={val_w[2]:.2f}, Occ={val_w[3]:.2f}"
            )

            # Log to CSV
            with open(csv_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"warmup,{epoch},{args.warmup_lr:.6f},{train_loss:.4f},0.0,0.0,0.0,0.0,"
                    f"{val_m['total_loss']:.4f},{val_m['loss_l1']:.4f},{val_m['ssim']:.4f},{val_m['psnr']:.2f},"
                    f"{val_m['loss_ce']:.4f},{val_m['loss_bal']:.4f},{val_m['gate_acc']:.4f},"
                    f"{val_w[0]:.4f},{val_w[1]:.4f},{val_w[2]:.4f},{val_w[3]:.4f}\n"
                )

            # Log to Excel
            excel_path = os.path.join(CURRENT_DIR, "task3_soft_moe_tracker.xlsx")
            append_warmup_epoch(
                epoch=epoch,
                lr=args.warmup_lr,
                train_loss=train_loss,
                val_loss=val_m["total_loss"],
                val_acc=val_m["gate_acc"],
                val_ssim=val_m["ssim"],
                weights_mean=val_w.tolist(),
                notes="Stage 1 Warmup",
                excel_path=excel_path,
            )

        # Save visual comparison after warm-up
        warmup_sample_path = os.path.join(args.samples_dir, "soft_moe_warmup_final.png")
        save_visual_samples(model, val_loader, device, warmup_sample_path, epoch_title="Stage 1 Gate Warm-Up")
        print(f"  --> Saved warm-up visual samples to: {warmup_sample_path}")

    # =========================================================================
    # STAGE 2: JOINT FINE-TUNING (ALL UNFROZEN)
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"STAGE 2: END-TO-END JOINT FINE-TUNING ({args.joint_epochs} EPOCHS, FULL SYSTEM)")
    print("=" * 75)
    model.unfreeze_experts()

    # Differential learning rates: smaller lr for experts, standard lr for gate if desired,
    # or unified small learning rate across all components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.joint_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.joint_epochs, eta_min=args.joint_lr * 0.05
    )

    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        train_accum = {
            "total_loss": 0.0, "loss_l1": 0.0, "loss_ssim": 0.0,
            "ssim": 0.0, "loss_ce": 0.0, "loss_bal": 0.0,
        }
        total_train = 0

        curr_lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(train_loader, desc=f"[Joint] Epoch {epoch:02d}/{args.joint_epochs:02d} (lr={curr_lr:.2e})")

        for corrupted, clean, labels, _ in pbar:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            reconstruction, weights, logits = model(corrupted)
            loss, breakdown = criterion(reconstruction, clean, logits, labels, weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            b = corrupted.size(0)
            total_train += b
            for k in train_accum:
                train_accum[k] += breakdown[k] * b

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ssim=f"{breakdown['ssim']:.4f}",
                ce=f"{breakdown['loss_ce']:.3f}",
            )

        scheduler.step()

        # Compute epoch training averages
        train_m = {k: v / total_train for k, v in train_accum.items()}

        # Evaluate on validation set
        val_m, val_w = evaluate_soft_moe(model, val_loader, criterion, device)

        is_best = val_m["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_m["total_loss"]
            best_val_ssim = val_m["ssim"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "temperature": model.temperature,
                    "val_loss": val_m["total_loss"],
                    "val_ssim": val_m["ssim"],
                    "val_psnr": val_m["psnr"],
                    "weights_mean": val_w.tolist(),
                    "config": vars(args),
                },
                best_ckpt_path,
            )

        print(
            f"[Joint Epoch {epoch:02d}] Train: Total={train_m['total_loss']:.4f}, SSIM={train_m['ssim']:.4f} | "
            f"Val: Total={val_m['total_loss']:.4f}, SSIM={val_m['ssim']:.4f}, PSNR={val_m['psnr']:.2f} dB | "
            f"Weights: [{val_w[0]:.2f}, {val_w[1]:.2f}, {val_w[2]:.2f}, {val_w[3]:.2f}]"
            f"{' [NEW BEST]' if is_best else ''}"
        )

        # Log to CSV
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"joint,{epoch},{curr_lr:.6f},{train_m['total_loss']:.4f},{train_m['loss_l1']:.4f},"
                f"{train_m['ssim']:.4f},{train_m['loss_ce']:.4f},{train_m['loss_bal']:.4f},"
                f"{val_m['total_loss']:.4f},{val_m['loss_l1']:.4f},{val_m['ssim']:.4f},{val_m['psnr']:.2f},"
                f"{val_m['loss_ce']:.4f},{val_m['loss_bal']:.4f},{val_m['gate_acc']:.4f},"
                f"{val_w[0]:.4f},{val_w[1]:.4f},{val_w[2]:.4f},{val_w[3]:.4f}\n"
            )

        # Log to Excel
        excel_path = os.path.join(CURRENT_DIR, "task3_soft_moe_tracker.xlsx")
        append_joint_epoch(
            epoch=epoch,
            lr=curr_lr,
            train_metrics=train_m,
            val_metrics=val_m,
            is_best=is_best,
            notes=f"New Best SSIM: {val_m['ssim']:.4f}" if is_best else "",
            excel_path=excel_path,
        )

        # Save visual comparison periodically
        if epoch % args.sample_interval == 0 or epoch == args.joint_epochs:
            sample_path = os.path.join(args.samples_dir, f"soft_moe_joint_epoch_{epoch:02d}.png")
            save_visual_samples(model, val_loader, device, sample_path, epoch_title=f"Joint Epoch {epoch:02d}")
            print(f"  --> Saved visual samples to: {sample_path}")

    print("\n" + "=" * 75)
    print("JOINT TRAINING COMPLETE!")
    print(f"Best Checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f}, Val SSIM: {best_val_ssim:.4f})")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Train Soft Mixture-of-Experts Restoration (Task 3)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Stage 1 gate warm-up epochs (experts frozen)")
    parser.add_argument("--joint_epochs", type=int, default=20, help="Stage 2 joint fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup_lr", type=float, default=5e-4, help="Learning rate for gate warm-up")
    parser.add_argument("--joint_lr", type=float, default=6e-5, help="Learning rate for joint fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax routing temperature tau")

    # Joint Loss Lambdas
    parser.add_argument("--lambda_rec", type=float, default=0.8, help="L1 reconstruction loss weight")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, help="SSIM structural similarity loss weight")
    parser.add_argument("--lambda_ce", type=float, default=0.1, help="Corruption classification loss weight")
    parser.add_argument("--lambda_bal", type=float, default=0.01, help="Routing balance regularizer weight")

    # Architecture
    parser.add_argument("--gate_base_channels", type=int, default=16, help="Gating network base channels")
    parser.add_argument("--gate_dropout", type=float, default=0.05, help="Gating network dropout")
    parser.add_argument("--specialist_base_channels", type=int, default=48, help="Specialist autoencoders base channels")
    parser.add_argument("--specialist_latent_dim", type=int, default=256, help="Specialist bottleneck dimension")

    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory of pre-trained Task 2 checkpoints")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save final model checkpoints")
    parser.add_argument("--samples_dir", type=str, default="MoE-restoration/visual_samples", help="Directory to save visual images")
    parser.add_argument("--sample_interval", type=int, default=5, help="Epoch interval to save visual samples")
    parser.add_argument("--config", type=str, default=None, help="Path to best_soft_moe_hyperparameters.json from Optuna")
    parser.add_argument("--scratch", action="store_true", help="Train from random scratch rather than pre-trained checkpoints")
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
            if "lambda_rec" in params and not hasattr(params, "lambda_ssim"):
                args.lambda_ssim = round(1.0 - float(params["lambda_rec"]), 4)
                print(f"  Applied lambda_ssim: {args.lambda_ssim} (1.0 - lambda_rec)")

    run_training_pipeline(args)


if __name__ == "__main__":
    main()
