"""
Task 3: Gating Network Behavioral Analysis & Routing Visualization
Report Requirements:
1. Average expert routing weights for every true corruption type and severity level (low, med, high)
2. Routing heatmap diagram (True Corruption vs Assigned Expert Weights)
3. Examples of dominant routing (one expert > 85%) vs distributed routing (weights shared across multiple experts)
4. Detection of inactive experts or domination of unrelated inputs
5. Full analysis exported to JSON, Excel Tracker, and high-res IEEE figures
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Path configuration
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECIALIST_DIR = os.path.join(WORKSPACE_ROOT, "specialist-autoencoders")
UNIVERSAL_DIR = os.path.join(WORKSPACE_ROOT, "universal-restoration")
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, SPECIALIST_DIR, UNIVERSAL_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from loader import (
    get_dataloaders,
    CORRUPTION_NAMES,
    LABEL_CLEAN,
    LABEL_SALT_PEPPER,
    LABEL_GAUSSIAN_BLUR,
    LABEL_OCCLUSION,
)
from soft_moe import SoftMoERestoration, BRANCH_SHORT_NAMES
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill


def analyze_routing_distribution(
    model: SoftMoERestoration,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates routing weights across corruption types and severities.
    Returns:
        matrix: (4, 4) mean weights matrix [true_class, expert_branch]
        metrics_dict: summary of dominance, entropy, and inactivity
        case_studies: gallery of dominant vs distributed samples
    """
    model.eval()

    # Per-class accumulators: 4 classes x 4 branches
    class_weight_sums = np.zeros((4, 4), dtype=np.float64)
    class_counts = np.zeros(4, dtype=np.int64)

    # Per-severity breakdown: key = (class_idx, severity_str)
    severity_weight_sums = {}
    severity_counts = {}

    all_samples = []

    with torch.no_grad():
        for corrupted, clean, labels, metas in val_loader:
            corrupted = corrupted.to(device)
            recon, weights, logits = model(corrupted)
            w_np = weights.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                c_idx = int(labels_np[i])
                w_vec = w_np[i]

                class_weight_sums[c_idx] += w_vec
                class_counts[c_idx] += 1

                # Extract severity if present in meta
                meta = metas[i] if isinstance(metas, list) and i < len(metas) else {}
                severity = meta.get("severity", "medium") if isinstance(meta, dict) else "medium"
                sev_key = f"{CORRUPTION_NAMES[c_idx]}_{severity}"

                if sev_key not in severity_weight_sums:
                    severity_weight_sums[sev_key] = np.zeros(4, dtype=np.float64)
                    severity_counts[sev_key] = 0

                severity_weight_sums[sev_key] += w_vec
                severity_counts[sev_key] += 1

                # Routing entropy: H = -sum(w * log(w))
                w_clean = np.clip(w_vec, 1e-8, 1.0)
                entropy = float(-np.sum(w_clean * np.log(w_clean)))
                max_w = float(np.max(w_vec))
                dom_expert = int(np.argmax(w_vec))

                all_samples.append({
                    "true_class": c_idx,
                    "true_name": CORRUPTION_NAMES[c_idx],
                    "weights": w_vec.tolist(),
                    "max_weight": max_w,
                    "entropy": entropy,
                    "dominant_expert": dom_expert,
                    "corrupted_tensor": corrupted[i].cpu(),
                    "clean_tensor": clean[i].cpu(),
                    "recon_tensor": recon[i].cpu(),
                })

    # Normalized 4x4 matrix
    mean_weights_matrix = np.zeros((4, 4), dtype=np.float64)
    for c in range(4):
        if class_counts[c] > 0:
            mean_weights_matrix[c] = class_weight_sums[c] / class_counts[c]

    # Inactivity and Domination analysis
    overall_mean = np.mean(mean_weights_matrix, axis=0)  # Average weight per expert
    inactive_experts = [BRANCH_SHORT_NAMES[i] for i, avg in enumerate(overall_mean) if avg < 0.05]
    dominant_experts = [BRANCH_SHORT_NAMES[i] for i, avg in enumerate(overall_mean) if avg > 0.50]

    # Find Top Dominant (low entropy, max weight > 0.90) and Top Distributed (high entropy, max weight < 0.50)
    sorted_by_max = sorted(all_samples, key=lambda s: s["max_weight"], reverse=True)
    sorted_by_entropy = sorted(all_samples, key=lambda s: s["entropy"], reverse=True)

    dominant_samples = sorted_by_max[:4]
    distributed_samples = sorted_by_entropy[:4]

    summary_metrics = {
        "mean_weights_matrix": mean_weights_matrix.tolist(),
        "overall_mean_expert_weights": overall_mean.tolist(),
        "class_counts": class_counts.tolist(),
        "inactive_experts": inactive_experts,
        "dominant_experts": dominant_experts,
        "severity_breakdown": {
            k: (severity_weight_sums[k] / max(severity_counts[k], 1)).tolist()
            for k in severity_weight_sums
        },
    }

    return mean_weights_matrix, summary_metrics, (dominant_samples, distributed_samples)


def plot_routing_heatmap(matrix: np.ndarray, save_path: str):
    """Generate high-contrast publication heatmap for IEEE report using matplotlib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    short_true = ["Clean", "Salt & Pepper", "Gaussian Blur", "Occlusion"]
    short_experts = ["Clean (Id)", "Noise Expert", "Blur Expert", "Occl Expert"]

    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Assigned Routing Weight w_k", fontsize=10, fontweight="bold")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(short_experts, fontsize=9, fontweight="bold")
    ax.set_yticklabels(short_true, fontsize=9, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    plt.title("Soft MoE: Gating Network Routing Weights Heatmap", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Assigned Expert Branch", fontsize=10, fontweight="bold")
    plt.ylabel("True Input Corruption Class", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[AnalyzeGating] Saved routing heatmap to: {save_path}")


def plot_case_studies(dominant_samples: List[Dict], distributed_samples: List[Dict], save_path: str):
    """Plot visual comparison of dominant single-expert routing vs distributed multi-expert routing."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), dpi=250)

    # 2 dominant samples + 2 distributed samples
    display_samples = dominant_samples[:2] + distributed_samples[:2]
    categories = ["DOMINANT (Single Expert)", "DOMINANT (Single Expert)", "DISTRIBUTED (Multi-Expert)", "DISTRIBUTED (Multi-Expert)"]

    for row_idx, sample in enumerate(display_samples):
        corrupted_img = sample["corrupted_tensor"].permute(1, 2, 0).numpy()
        clean_img = sample["clean_tensor"].permute(1, 2, 0).numpy()
        recon_img = sample["recon_tensor"].permute(1, 2, 0).numpy()
        w = sample["weights"]

        # Column 0: Corrupted
        axes[row_idx, 0].imshow(np.clip(corrupted_img, 0, 1))
        axes[row_idx, 0].set_title(f"Input ({sample['true_name']})\n[{categories[row_idx]}]", fontsize=9)
        axes[row_idx, 0].axis("off")

        # Column 1: Clean Target
        axes[row_idx, 1].imshow(np.clip(clean_img, 0, 1))
        axes[row_idx, 1].set_title("Ground Truth Clean", fontsize=9)
        axes[row_idx, 1].axis("off")

        # Column 2: Reconstructed
        axes[row_idx, 2].imshow(np.clip(recon_img, 0, 1))
        axes[row_idx, 2].set_title("Soft MoE Restoration", fontsize=9)
        axes[row_idx, 2].axis("off")

        # Column 3: Routing Weights Bar Chart
        bar_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#9B59B6"]
        axes[row_idx, 3].bar(BRANCH_SHORT_NAMES, w, color=bar_colors, edgecolor="black", width=0.6)
        axes[row_idx, 3].set_ylim(0.0, 1.05)
        axes[row_idx, 3].set_title(f"Weights (H={sample['entropy']:.2f})", fontsize=9)
        for i, val in enumerate(w):
            axes[row_idx, 3].text(i, val + 0.03, f"{val:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[AnalyzeGating] Saved dominant vs distributed case studies to: {save_path}")


def log_to_excel_tracker(summary: Dict[str, Any], excel_path: str):
    """Update Routing_Distribution sheet in task3_soft_moe_tracker.xlsx."""
    if not os.path.exists(excel_path):
        return

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Routing_Distribution"]

    # Clear previous data rows (keep headers on row 3)
    while ws.max_row > 3:
        ws.delete_rows(4)

    matrix = summary["mean_weights_matrix"]
    for c_idx, c_name in enumerate(CORRUPTION_NAMES):
        w = matrix[c_idx]
        dom_idx = int(np.argmax(w))
        w_clean = np.clip(w, 1e-8, 1.0)
        entropy = float(-np.sum(w_clean * np.log(w_clean)))
        status = "Healthy" if w[c_idx] > 0.70 else "Distributed"

        ws.append([
            c_name,
            "Aggregate",
            round(w[0], 4),
            round(w[1], 4),
            round(w[2], 4),
            round(w[3], 4),
            BRANCH_SHORT_NAMES[dom_idx],
            round(entropy, 3),
            status,
        ])

    # Severity Breakdown rows
    sev_dict = summary.get("severity_breakdown", {})
    for key, w in sev_dict.items():
        parts = key.split("_")
        c_name = "_".join(parts[:-1])
        sev = parts[-1]
        dom_idx = int(np.argmax(w))
        w_clean = np.clip(w, 1e-8, 1.0)
        entropy = float(-np.sum(w_clean * np.log(w_clean)))

        ws.append([
            c_name,
            sev,
            round(w[0], 4),
            round(w[1], 4),
            round(w[2], 4),
            round(w[3], 4),
            BRANCH_SHORT_NAMES[dom_idx],
            round(entropy, 3),
            "Severity Breakdown",
        ])

    wb.save(excel_path)
    print(f"[AnalyzeGating] Logged routing distribution to: {excel_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Soft MoE Gating Routing")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_soft_moe.pt")
    parser.add_argument("--output_json", type=str, default="checkpoints/soft_moe_gating_analysis.json")
    parser.add_argument("--heatmap_path", type=str, default="MoE-restoration/visual_samples/gating_routing_heatmap.png")
    parser.add_argument("--cases_path", type=str, default="MoE-restoration/visual_samples/dominant_vs_distributed_routing.png")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[AnalyzeGating] Loading dataset from: {WORKSPACE_ROOT}")

    manifest_dir = os.path.join(WORKSPACE_ROOT, "manifests")
    _, val_loader, _ = get_dataloaders(
        base_dir=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        num_workers=0,
        manifest_dir=manifest_dir,
        seed=args.seed,
    )

    # Initialize model
    model = SoftMoERestoration().to(device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"[AnalyzeGating] Loaded trained weights from: {args.model_path}")
    else:
        print(f"[AnalyzeGating] Notice: Model checkpoint not found at {args.model_path}. Loading Task 2 pre-trained weights for initial analysis.")
        model.load_pretrained_weights(
            classifier_path=os.path.join("checkpoints", "best_corruption_classifier.pt"),
            noise_path=os.path.join("checkpoints", "best_specialist_noise.pt"),
            blur_path=os.path.join("checkpoints", "best_specialist_blur.pt"),
            occlusion_path=os.path.join("checkpoints", "best_specialist_occlusion.pt"),
            device=device,
        )

    print("[AnalyzeGating] Running routing distribution analysis...")
    matrix, summary, (dominant, distributed) = analyze_routing_distribution(model, val_loader, device)

    # Save Heatmap & Case Studies
    plot_routing_heatmap(matrix, args.heatmap_path)
    plot_case_studies(dominant, distributed, args.cases_path)

    # Save Metrics JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[AnalyzeGating] Saved metrics summary JSON to: {args.output_json}")

    # Save to Excel
    excel_path = os.path.join(CURRENT_DIR, "task3_soft_moe_tracker.xlsx")
    log_to_excel_tracker(summary, excel_path)

    print("\n" + "=" * 65)
    print("ROUTING ANALYSIS COMPLETE")
    print(f"Overall Expert Inactivity: {summary['inactive_experts'] or 'None (All experts active)'}")
    print(f"Dominant Experts:          {summary['dominant_experts'] or 'None (Balanced routing)'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
