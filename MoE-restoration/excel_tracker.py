"""
Task 3: Excel Training Tracker & Real-Time Logger
Creates and maintains 'task3_soft_moe_tracker.xlsx' with comprehensive tracking:
1. Overview_Summary: Architecture details, loss configuration, hyperparameters, best scores
2. Warmup_Gate_Training: Stage 1 frozen-expert warm-up progression
3. Joint_Finetuning: Stage 2 end-to-end joint fine-tuning metrics
4. Routing_Distribution: Per-corruption average routing weights and distribution analysis
5. Optuna_Study: Hyperparameter search trials and pruning log
"""

import os
from typing import Dict, Any, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "task3_soft_moe_tracker.xlsx")


def create_styled_task3_tracker(output_path: str = EXCEL_PATH):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    # Color Palette & Styling
    navy_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    dark_teal_fill = PatternFill(start_color="0E6655", end_color="0E6655", fill_type="solid")
    purple_fill = PatternFill(start_color="5B2C6F", end_color="5B2C6F", fill_type="solid")
    slate_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    indigo_fill = PatternFill(start_color="283747", end_color="283747", fill_type="solid")

    title_font = Font(name="Segoe UI", size=14, bold=True, color="1B365D")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    bold_font = Font(name="Segoe UI", size=10, bold=True)
    regular_font = Font(name="Segoe UI", size=10)

    thin_border = Border(
        left=Side(style='thin', color='BDC3C7'),
        right=Side(style='thin', color='BDC3C7'),
        top=Side(style='thin', color='BDC3C7'),
        bottom=Side(style='thin', color='BDC3C7'),
    )
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")

    # =========================================================================
    # SHEET 1: Overview_Summary
    # =========================================================================
    ws_ov = wb.create_sheet(title="Overview_Summary")
    ws_ov.views.sheetView[0].showGridLines = True
    ws_ov.merge_cells("A1:G1")
    ws_ov["A1"] = "Task 3: Differentiable Soft Mixture-of-Experts (MoE) Restoration Summary"
    ws_ov["A1"].font = title_font

    ws_ov.cell(row=3, column=1, value="System Configuration").font = bold_font
    overview_rows = [
        ("Architecture", "Soft Mixture-of-Experts (Gating CNN + 3 Specialist Autoencoders + Clean Identity Bypass)"),
        ("Gating Network", "4-Stage ConvNet with Global Average Pooling (Init: Task 2 best_corruption_classifier.pt)"),
        ("Experts", "Noise (Salt & Pepper), Blur (Gaussian), Occlusion (Inpainting) from Task 2 checkpoints"),
        ("Routing Equation", "w = softmax( G(x) / tau ) across 4 branches"),
        ("Reconstruction", "x_hat = w[0]*x + w[1]*A_salt(x) + w[2]*A_blur(x) + w[3]*A_occlusion(x)"),
        ("Loss Formulation", "L_joint = lambda_rec * L1 + lambda_ssim * (1-SSIM) + lambda_ce * L_ce + lambda_bal * L_bal"),
        ("Routing Balance Regularizer", "L_bal = sum_{k=1}^4 (w_bar_k - 0.25)^2 to prevent expert collapse"),
        ("Training Strategy", "2-Stage: Stage 1 Warm-up (Experts frozen) -> Stage 2 Joint Fine-Tuning (All unfrozen)"),
    ]
    for r_idx, (label, val) in enumerate(overview_rows, start=4):
        ws_ov.cell(row=r_idx, column=1, value=label).font = bold_font
        ws_ov.cell(row=r_idx, column=2, value=val).font = regular_font

    # =========================================================================
    # SHEET 2: Warmup_Gate_Training
    # =========================================================================
    ws_warm = wb.create_sheet(title="Warmup_Gate_Training")
    ws_warm.views.sheetView[0].showGridLines = True
    ws_warm.merge_cells("A1:K1")
    ws_warm["A1"] = "Stage 1: Gating Network Warm-Up (Experts Frozen)"
    ws_warm["A1"].font = title_font

    warm_headers = [
        "Epoch", "LR", "Train Loss", "Val Loss", "Val Acc (%)",
        "Val SSIM", "Clean w0", "Noise w1", "Blur w2", "Occlusion w3", "Notes"
    ]
    for col_idx, h in enumerate(warm_headers, start=1):
        c = ws_warm.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = dark_teal_fill
        c.alignment = center_align

    # =========================================================================
    # SHEET 3: Joint_Finetuning
    # =========================================================================
    ws_joint = wb.create_sheet(title="Joint_Finetuning")
    ws_joint.views.sheetView[0].showGridLines = True
    ws_joint.merge_cells("A1:O1")
    ws_joint["A1"] = "Stage 2: End-to-End Joint Fine-Tuning (Gate + All Experts Unfrozen)"
    ws_joint["A1"].font = title_font

    joint_headers = [
        "Epoch", "LR", "Train Total", "Train L1", "Train SSIM", "Train CE", "Train Bal",
        "Val Total", "Val L1", "Val SSIM", "Val PSNR (dB)", "Val CE", "Val Bal",
        "Best Model?", "Observations"
    ]
    for col_idx, h in enumerate(joint_headers, start=1):
        c = ws_joint.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = navy_fill
        c.alignment = center_align

    # =========================================================================
    # SHEET 4: Routing_Distribution
    # =========================================================================
    ws_rout = wb.create_sheet(title="Routing_Distribution")
    ws_rout.views.sheetView[0].showGridLines = True
    ws_rout.merge_cells("A1:I1")
    ws_rout["A1"] = "Gating Network Analysis: Average Routing Weights per True Corruption & Severity"
    ws_rout["A1"].font = title_font

    rout_headers = [
        "True Corruption", "Severity Level", "Mean Clean (w0)", "Mean Noise (w1)",
        "Mean Blur (w2)", "Mean Occlusion (w3)", "Dominant Branch", "Routing Entropy", "Status"
    ]
    for col_idx, h in enumerate(rout_headers, start=1):
        c = ws_rout.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = purple_fill
        c.alignment = center_align

    # =========================================================================
    # SHEET 5: Optuna_Study
    # =========================================================================
    ws_opt = wb.create_sheet(title="Optuna_Study")
    ws_opt.views.sheetView[0].showGridLines = True
    ws_opt.merge_cells("A1:L1")
    ws_opt["A1"] = "Task 3 Optuna Hyperparameter Optimization Study"
    ws_opt["A1"].font = title_font

    opt_headers = [
        "Trial", "Joint LR", "Tau (Temp)", "Lambda CE", "Lambda Bal", "Lambda Rec",
        "Lambda SSIM", "Val SSIM", "Val PSNR (dB)", "Val Loss", "Routing Status", "Pruned?"
    ]
    for col_idx, h in enumerate(opt_headers, start=1):
        c = ws_opt.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = slate_fill
        c.alignment = center_align

    # Auto-adjust column widths for all sheets
    for ws in [ws_ov, ws_warm, ws_joint, ws_rout, ws_opt]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if len(val) > max_len and "\n" not in val and len(val) < 80:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)
    print(f"[ExcelTracker] Initialized Task 3 workbook at: {output_path}")
    return wb


def append_warmup_epoch(
    epoch: int,
    lr: float,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    val_ssim: float,
    weights_mean: list,
    notes: str = "",
    excel_path: str = EXCEL_PATH,
):
    """Log an epoch of Stage 1 Warm-up."""
    if not os.path.exists(excel_path):
        create_styled_task3_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Warmup_Gate_Training"]
    next_row = ws.max_row + 1

    row_data = [
        epoch,
        f"{lr:.6f}",
        round(train_loss, 4),
        round(val_loss, 4),
        round(val_acc * 100.0, 2),
        round(val_ssim, 4),
        round(weights_mean[0], 4) if len(weights_mean) > 0 else 0.0,
        round(weights_mean[1], 4) if len(weights_mean) > 1 else 0.0,
        round(weights_mean[2], 4) if len(weights_mean) > 2 else 0.0,
        round(weights_mean[3], 4) if len(weights_mean) > 3 else 0.0,
        notes,
    ]
    for c_idx, val in enumerate(row_data, start=1):
        cell = ws.cell(row=next_row, column=c_idx, value=val)
        cell.font = Font(name="Segoe UI", size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(excel_path)


def append_joint_epoch(
    epoch: int,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_best: bool = False,
    notes: str = "",
    excel_path: str = EXCEL_PATH,
):
    """Log an epoch of Stage 2 Joint Fine-Tuning."""
    if not os.path.exists(excel_path):
        create_styled_task3_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Joint_Finetuning"]
    next_row = ws.max_row + 1

    green_fill = PatternFill(start_color="D4EFDF", end_color="D4EFDF", fill_type="solid")

    row_data = [
        epoch,
        f"{lr:.6f}",
        round(train_metrics.get("total_loss", 0.0), 4),
        round(train_metrics.get("loss_l1", 0.0), 4),
        round(train_metrics.get("ssim", 0.0), 4),
        round(train_metrics.get("loss_ce", 0.0), 4),
        round(train_metrics.get("loss_bal", 0.0), 4),
        round(val_metrics.get("total_loss", 0.0), 4),
        round(val_metrics.get("loss_l1", 0.0), 4),
        round(val_metrics.get("ssim", 0.0), 4),
        round(val_metrics.get("psnr", 0.0), 2),
        round(val_metrics.get("loss_ce", 0.0), 4),
        round(val_metrics.get("loss_bal", 0.0), 4),
        "YES" if is_best else "No",
        notes,
    ]
    for c_idx, val in enumerate(row_data, start=1):
        cell = ws.cell(row=next_row, column=c_idx, value=val)
        cell.font = Font(name="Segoe UI", size=10, bold=is_best)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        if is_best:
            cell.fill = green_fill

    wb.save(excel_path)


def append_optuna_trial(
    trial_num: int,
    params: Dict[str, Any],
    val_ssim: float,
    val_psnr: float,
    val_loss: float,
    routing_status: str,
    pruned: bool,
    excel_path: str = EXCEL_PATH,
):
    """Log an Optuna trial result."""
    if not os.path.exists(excel_path):
        create_styled_task3_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Optuna_Study"]
    next_row = ws.max_row + 1

    row_data = [
        trial_num,
        f"{params.get('lr', 0.0):.6f}",
        round(params.get("tau", 1.0), 3),
        round(params.get("lambda_ce", 0.1), 3),
        round(params.get("lambda_bal", 0.01), 4),
        round(params.get("lambda_rec", 0.8), 3),
        round(params.get("lambda_ssim", 0.2), 3),
        round(val_ssim, 4) if val_ssim is not None else "N/A",
        round(val_psnr, 2) if val_psnr is not None else "N/A",
        round(val_loss, 4) if val_loss is not None else "N/A",
        routing_status,
        "YES (Pruned)" if pruned else "Completed",
    ]
    for c_idx, val in enumerate(row_data, start=1):
        cell = ws.cell(row=next_row, column=c_idx, value=val)
        cell.font = Font(name="Segoe UI", size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(excel_path)


if __name__ == "__main__":
    create_styled_task3_tracker()
