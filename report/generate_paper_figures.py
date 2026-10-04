"""
Script to generate all publication-grade figures for the IEEE Technical Report.
Reads real training logs, Optuna study records, and evaluation metrics from checkpoints/.
Outputs high-resolution 300 DPI figures to report/figures/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def generate_confusion_matrix_plot():
    json_path = os.path.join(
        WORKSPACE_ROOT, "evaluation_results", "tasks_1_3", "full_official_test_results.json"
    )
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    raw_cm = np.array(data["overall"]["task2_hard_predicted"]["confusion_matrix"])
    cm = raw_cm / raw_cm.sum(axis=1, keepdims=True) * 100.0
    accuracy = np.trace(raw_cm) / raw_cm.sum() * 100.0
    labels = ["Clean", "Noise\n(S&P)", "Blur\n(Gauss)", "Occlusion\n(Box)"]

    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%\n({raw_cm[i, j]})",
                    ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Corruption Class", fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontweight="bold")
    ax.set_title(
        f"Task 2: Official-Test Routing Confusion Matrix\n(Accuracy: {accuracy:.2f}%)",
        fontsize=10.5,
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Classification Rate (%)", rotation=-90, va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fig_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {save_path}")


def generate_gating_heatmap_plot():
    json_path = os.path.join(
        WORKSPACE_ROOT, "evaluation_results", "tasks_1_3", "full_official_test_results.json"
    )
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    manifests = data["results_by_manifest"]
    groups = [
        [manifests["clean"]],
        [manifests[f"salt_and_pepper_{s}"] for s in ("low", "medium", "high")],
        [manifests[f"gaussian_blur_{s}"] for s in ("low", "medium", "high")],
        [manifests[f"rectangular_occlusion_{s}"] for s in ("low", "medium", "high")],
    ]
    matrix = np.array([
        np.mean([row["systems"]["task3_soft_moe"]["mean_routing_weights"] for row in group], axis=0)
        for group in groups
    ]) * 100.0
    inputs = ["Clean Input", "Noise Input", "Blur Input", "Occlusion Input"]
    experts = ["Clean\nBypass", "Noise\nSpecialist", "Blur\nSpecialist", "Occlusion\nSpecialist"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Subplot 1: Matrix Heatmap
    im = ax1.imshow(matrix, cmap="Purples", vmin=0, vmax=100)
    for i in range(len(inputs)):
        for j in range(len(experts)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

    ax1.set_xticks(range(len(experts)))
    ax1.set_yticks(range(len(inputs)))
    ax1.set_xticklabels(experts)
    ax1.set_yticklabels(inputs)
    ax1.set_title("(a) Mean Gating Weights $w_k(x)$ per Corruption", fontsize=10, fontweight="bold")

    # Subplot 2: Overall Load Balance
    balance = [w * 100.0 for w in data["overall"]["task3_soft_moe"]["mean_routing_weights"]]
    bars = ax2.bar(range(4), balance, color=["#4A148C", "#7B1FA2", "#9C27B0", "#BA68C8"], width=0.55, edgecolor="black", linewidth=0.8)
    expected = [10.0, 30.0, 30.0, 30.0]
    ax2.scatter(range(4), expected, color="red", marker="_", s=220, linewidths=2,
                label="Manifest share")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["Clean", "Noise", "Blur", "Occl"], fontsize=9)
    ax2.set_ylabel("Global Utilization (%)", fontsize=10)
    ax2.set_ylim(0, 35)
    ax2.set_title("(b) Overall Weights vs. Manifest Share", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fig_moe_gating_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {save_path}")


def generate_training_curves_plot():
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), dpi=300)

    # 1. Task 1: Universal Autoencoder
    t1_path = os.path.join(WORKSPACE_ROOT, "universal-restoration", "task1_training_tracker.csv")
    if os.path.exists(t1_path):
        df1 = pd.read_csv(t1_path)
        axes[0, 0].plot(df1["epoch"], df1["train_loss"], label="Train Loss", color="#1F77B4", linewidth=1.5)
        axes[0, 0].plot(df1["epoch"], df1["val_loss"], label="Val Loss", color="#FF7F0E", linewidth=1.5, linestyle="--")
        axes[0, 0].set_title("(a) Task 1: Universal Autoencoder Loss", fontsize=10, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Reconstruction Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Task 2: Specialists SSIM Progression
    n_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "training_log_noise.csv")
    b_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "training_log_blur.csv")
    o_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "training_log_occlusion.csv")
    if os.path.exists(n_path) and os.path.exists(b_path) and os.path.exists(o_path):
        df_n = pd.read_csv(n_path)
        df_b = pd.read_csv(b_path)
        df_o = pd.read_csv(o_path)
        axes[0, 1].plot(df_n["epoch"], df_n["val_ssim"], label="Noise Spec", color="#2CA02C", linewidth=1.4)
        axes[0, 1].plot(df_b["epoch"], df_b["val_ssim"], label="Blur Spec", color="#1F77B4", linewidth=1.4)
        axes[0, 1].plot(df_o["epoch"], df_o["val_ssim"], label="Occlusion Spec", color="#D62728", linewidth=1.4)
        axes[0, 1].set_title("(b) Task 2: Specialist Autoencoders (Val SSIM)", fontsize=10, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Validation SSIM")
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Task 3: Soft MoE Loss
    moe_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "soft_moe_training_log.csv")
    if os.path.exists(moe_path):
        df_m = pd.read_csv(moe_path)
        # Filter for joint fine-tuning stage or overall progression
        joint_df = df_m[df_m["stage"] == "joint"] if "stage" in df_m.columns else df_m
        axes[1, 0].plot(range(1, len(joint_df) + 1), joint_df["train_total_loss"], label="Train Total Loss", color="#9467BD", linewidth=1.5)
        axes[1, 0].plot(range(1, len(joint_df) + 1), joint_df["val_total_loss"], label="Val Total Loss", color="#8C564B", linewidth=1.5, linestyle="--")
        axes[1, 0].set_title("(c) Task 3: Soft MoE Joint Optimization", fontsize=10, fontweight="bold")
        axes[1, 0].set_xlabel("Joint Fine-Tuning Epoch")
        axes[1, 0].set_ylabel("Joint Multi-Task Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Task 4: cGAN Loss & Val SSIM
    cgan_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "cgan_training_log.csv")
    if os.path.exists(cgan_path):
        df_c = pd.read_csv(cgan_path)
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        l1 = ax4.plot(df_c["epoch"], df_c["d_total"], label="Discriminator Loss", color="#D62728", linewidth=1.2, alpha=0.85)
        l2 = ax4.plot(df_c["epoch"], df_c["g_adv"], label="Generator Adv Loss", color="#1F77B4", linewidth=1.2, alpha=0.85)
        l3 = ax4_twin.plot(df_c["epoch"], df_c["val_ssim"], label="Val SSIM", color="#2CA02C", linewidth=1.6)

        ax4.set_title("(d) Task 4: Pix2Pix cGAN Convergence", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss Magnitude")
        ax4_twin.set_ylabel("Validation SSIM", color="#2CA02C", fontweight="bold")
        ax4_twin.tick_params(axis="y", labelcolor="#2CA02C")

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper right")
        ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fig_training_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {save_path}")


def generate_optuna_study_plot():
    tracker_path = os.path.join(WORKSPACE_ROOT, "face-to-sketch Generator", "task4_gan_tracker.xlsx")
    selected_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_cgan_hyperparameters.json")
    if not os.path.exists(tracker_path) or not os.path.exists(selected_path):
        return

    df = pd.read_excel(tracker_path, sheet_name="Optuna_Study", header=2)
    df = df[pd.to_numeric(df["Trial"], errors="coerce").notna()].copy()
    with open(selected_path, "r", encoding="utf-8") as f:
        selected = json.load(f)
    trials_data = [
        {
            "trial": int(row["Trial"]),
            "lambda": float(row["Lambda L1"]),
            "ssim": float(row["Val SSIM"]),
            "pruned": str(row["Pruned?"]).strip().lower().startswith("yes"),
        }
        for _, row in df.iterrows()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=300)

    # Subplot 1: Trial Progression and Pruning
    completed = [t for t in trials_data if not t["pruned"]]
    pruned = [t for t in trials_data if t["pruned"]]

    ax1.scatter([t["trial"] for t in completed], [t["ssim"] for t in completed],
                color="#2CA02C", s=60, label="Completed Trial", edgecolors="black", zorder=3)
    ax1.scatter([t["trial"] for t in pruned], [t["ssim"] for t in pruned],
                color="#D62728", marker="x", s=50, label="Pruned (MedianPruner)", zorder=3)

    selected_trial = int(selected["best_trial_number"])
    best_t = next(t for t in trials_data if t["trial"] == selected_trial)
    best_t["ssim"] = float(selected["best_val_ssim"])
    ax1.annotate(f"Selected: Trial #{best_t['trial']}\n(SSIM: {best_t['ssim']:.4f})",
                 xy=(best_t["trial"], best_t["ssim"]),
                 xytext=(best_t["trial"] - 3.5, best_t["ssim"] - 0.12),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                 fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D"))

    ax1.set_xlabel("Optuna Trial Number")
    ax1.set_ylabel("Validation SSIM")
    ax1.set_title("(a) Optuna Optimization Progression", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Subplot 2: Lambda L1 vs Validation SSIM
    lambdas = [t["lambda"] for t in trials_data]
    ssims = [t["ssim"] for t in trials_data]
    colors = ["#D62728" if t["pruned"] else "#2CA02C" for t in trials_data]

    ax2.scatter(lambdas, ssims, c=colors, s=65, edgecolors="black", alpha=0.85, zorder=3)
    ax2.set_xlabel(r"Reconstruction Loss Weight ($\lambda_{L1}$)")
    ax2.set_ylabel("Validation SSIM")
    ax2.set_title(r"(b) Sensitivity to $\lambda_{L1}$ Weight", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)


    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fig_optuna_study.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {save_path}")


def copy_visual_results():
    # Copy test_qualitative_grid.png
    src_grid = os.path.join(WORKSPACE_ROOT, "face-to-sketch Generator", "visual_samples", "test_qualitative_grid.png")
    dst_grid = os.path.join(FIGURES_DIR, "fig_cgan_qualitative_grid.png")
    if os.path.exists(src_grid):
        im = Image.open(src_grid)
        im.save(dst_grid)
        print(f"Copied: {dst_grid}")

    # Copy Task 3 Soft MoE visual comparison
    src_moe = os.path.join(WORKSPACE_ROOT, "MoE-restoration", "visual_samples", "soft_moe_joint_epoch_20.png")
    dst_moe = os.path.join(FIGURES_DIR, "fig_moe_visual_samples.png")
    if os.path.exists(src_moe):
        im = Image.open(src_moe)
        im.save(dst_moe)
        print(f"Copied: {dst_moe}")

    # Copy Task 1 Universal restoration sample
    src_t1 = os.path.join(WORKSPACE_ROOT, "universal-restoration", "visual_samples", "epoch_50_restoration.png")
    dst_t1 = os.path.join(FIGURES_DIR, "fig_universal_restoration.png")
    if os.path.exists(src_t1):
        im = Image.open(src_t1)
        im.save(dst_t1)
        print(f"Copied: {dst_t1}")


def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                     facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=7.5, color="#333333")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1B365D", lw=1.5, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center", va="bottom", fontsize=7.5, color="#1B365D", fontweight="semibold")

    # Inputs
    draw_box(0.02, 0.55, 0.16, 0.35, "Input Image $x$", "Pet / Human Face\n(128x128x3)", "#E8F0FE", "#1A73E8")
    draw_box(0.02, 0.10, 0.16, 0.35, "Style Code $s$", "Category Index\n$s \\in \\{0, 1, 2\\}$", "#E8F0FE", "#1A73E8")

    # Task 1
    draw_box(0.24, 0.72, 0.22, 0.24, "Task 1: Universal AE", "Shared Bottleneck\n$E_{uni} \\to D_{uni}$", "#E8F8F5", "#117A65")
    draw_arrow(0.18, 0.75, 0.24, 0.84)

    # Task 2
    draw_box(0.24, 0.42, 0.22, 0.24, "Task 2: Hard Router", "4-Class Classifier\n+ 3 Specialists", "#FEF9E7", "#B7950B")
    draw_arrow(0.18, 0.68, 0.24, 0.54)

    # Task 3
    draw_box(0.24, 0.12, 0.22, 0.24, "Task 3: Soft MoE", "Softmax Gating $w_k(x)$\nJoint Differentiable", "#F4ECF7", "#7D3C98")
    draw_arrow(0.18, 0.62, 0.24, 0.24)

    # Task 4
    draw_box(0.53, 0.25, 0.24, 0.52, "Task 4: cGAN (Pix2Pix)", "U-Net Generator (16.7M)\nSkip Conns + Style Emb\n70x70 PatchGAN (2.8M)", "#FADBD8", "#B03A2E")
    draw_arrow(0.18, 0.58, 0.53, 0.60, label="Photo $x$")
    draw_arrow(0.18, 0.28, 0.53, 0.38, label="Style $s$")

    # Deployment
    draw_box(0.82, 0.25, 0.16, 0.52, "Deployment Core", "ONNX Runtime C++\nFastAPI REST Engine\nReact 19 UI + Nginx", "#D5F5E3", "#196F3D")
    draw_arrow(0.46, 0.84, 0.82, 0.65)
    draw_arrow(0.46, 0.54, 0.82, 0.54)
    draw_arrow(0.46, 0.24, 0.82, 0.42)
    draw_arrow(0.77, 0.51, 0.82, 0.51)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Unified Architecture: Image Restoration (Tasks 1-3) & Face-to-Sketch Synthesis (Task 4)", fontsize=11, fontweight="bold", pad=8)

    save_path = os.path.join(FIGURES_DIR, "fig_pipeline_architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {save_path}")


def generate_app_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                     facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=7.2, color="#333333")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1B365D", lw=1.5, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center", va="bottom", fontsize=7.2, color="#1B365D", fontweight="semibold")

    # Layer 1: Client Browser / Frontend
    draw_box(0.02, 0.20, 0.22, 0.65, "Client Interface\n(React 19 + Tailwind)", "• Preset Pet & Face Gallery\n• Drag-and-Drop Image Upload\n• Real-Time HTML5 Webcam\n• Multi-Style Switcher & Download", "#E8F0FE", "#1A73E8")

    # Layer 2: Nginx Reverse Proxy
    draw_box(0.28, 0.35, 0.16, 0.35, "Reverse Proxy\n(Nginx Alpine)", "• Static Asset Caching\n• /api & /health Routing\n• Host Port 5173:80", "#FEF9E7", "#B7950B")
    draw_arrow(0.24, 0.525, 0.28, 0.525, label="HTTP/JSON")

    # Layer 3: Backend API (FastAPI)
    draw_box(0.48, 0.15, 0.22, 0.75, "Application Gateway\n(FastAPI / Uvicorn)", "• Endpoints: /api/face-to-sketch,\n  /api/universal-restoration,\n  /api/hard-routing, /api/soft-mixture\n• Dynamic Batch Tensor Preprocessing\n• Base64 Encoding & Validation", "#F4ECF7", "#7D3C98")
    draw_arrow(0.44, 0.525, 0.48, 0.525, label="ProxyPass")

    # Layer 4: ONNX Runtime Engine Core
    draw_box(0.74, 0.15, 0.24, 0.75, "ONNX Runtime\nInference", "• Universal AE\n• Corruption classifier\n• 3 specialist autoencoders\n• Soft MoE\n• cGAN generator", "#D5F5E3", "#196F3D")
    draw_arrow(0.70, 0.525, 0.74, 0.525, label="Direct C++ API")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Repository Application and Inference Architecture", fontsize=10.5, fontweight="bold", pad=8)

    save_path = os.path.join(FIGURES_DIR, "fig_app_architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {save_path}")


if __name__ == "__main__":
    generate_confusion_matrix_plot()
    generate_gating_heatmap_plot()
    generate_training_curves_plot()
    generate_optuna_study_plot()
    generate_pipeline_diagram()
    # Genuine held-out qualitative/failure figures are generated by
    # scripts/generate_report_visual_evidence.py and intentionally left intact.
    generate_app_architecture_diagram()
    copy_visual_results()
    print("All figures successfully created in report/figures/!")
