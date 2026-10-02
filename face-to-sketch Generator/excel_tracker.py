"""
Task 4: Excel Training Tracker & Real-Time Logger
Creates and maintains 'task4_gan_tracker.xlsx' with comprehensive tracking:
1. Overview_Summary: Architecture details, GAN loss formulation, Optuna hyperparameters
2. Training_Progress: Epoch-by-epoch D_loss (real/fake), G_loss (adv/L1), and Val metrics (SSIM, PSNR, L1)
3. Style_Evaluation: Per-style breakdown (Style 1, Style 2, Style 3 metrics)
4. Optuna_Study: Hyperparameter search trials and pruning log
"""

import os
from typing import Dict, Any, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "task4_gan_tracker.xlsx")


def create_styled_task4_tracker(output_path: str = EXCEL_PATH):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    # Color Palette & Styling
    navy_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    dark_teal_fill = PatternFill(start_color="0E6655", end_color="0E6655", fill_type="solid")
    crimson_fill = PatternFill(start_color="78281F", end_color="78281F", fill_type="solid")
    slate_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    purple_fill = PatternFill(start_color="512E5F", end_color="512E5F", fill_type="solid")

    title_font = Font(name="Segoe UI", size=14, bold=True, color="1B365D")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    bold_font = Font(name="Segoe UI", size=10, bold=True)
    regular_font = Font(name="Segoe UI", size=10)

    center_align = Alignment(horizontal="center", vertical="center")

    # =========================================================================
    # SHEET 1: Overview_Summary
    # =========================================================================
    ws_ov = wb.create_sheet(title="Overview_Summary")
    ws_ov.views.sheetView[0].showGridLines = True
    ws_ov.merge_cells("A1:G1")
    ws_ov["A1"] = "Task 4: Conditional GAN Face-to-Sketch Synthesis Summary"
    ws_ov["A1"].font = title_font

    ws_ov.cell(row=3, column=1, value="System Specifications").font = bold_font
    overview_rows = [
        ("Task Name", "Style-Conditioned Face-to-Sketch Synthesis (Conditional GAN)"),
        ("Dataset", "FS2K Dataset (2,104 paired facial photographs and sketches)"),
        ("Styles", "3 Sketch Styles: Style 1 (Shaded Pencil), Style 2 (Clean Contour), Style 3 (Graphic Comic)"),
        ("Dataset Split", "Official Train (1,058): 899 Train (85%) / 159 Val (15% Stratified by Style, seed 42)"),
        ("Held-out Test", "Official Test (1,046 pairs) held out strictly for final evaluation"),
        ("Generator Architecture", "U-Net Encoder-Decoder (5 down / 4 up) with skip connections & spatial style injection"),
        ("Discriminator Architecture", "70x70 PatchGAN local patch realism classifier with style conditioning"),
        ("Style Representation", "Learned categorical embedding (nn.Embedding) spatially broadcast to 128x128"),
        ("Loss Formulation", "L_G = L_adv + lambda_L1 * L_L1 (default lambda_L1 = 100.0) | L_D = 0.5 * (L_real + L_fake)"),
    ]
    for r_idx, (label, val) in enumerate(overview_rows, start=4):
        ws_ov.cell(row=r_idx, column=1, value=label).font = bold_font
        ws_ov.cell(row=r_idx, column=2, value=val).font = regular_font

    # =========================================================================
    # SHEET 2: Training_Progress
    # =========================================================================
    ws_train = wb.create_sheet(title="Training_Progress")
    ws_train.views.sheetView[0].showGridLines = True
    ws_train.merge_cells("A1:N1")
    ws_train["A1"] = "Conditional GAN Epoch Training Progression"
    ws_train["A1"].font = title_font

    train_headers = [
        "Epoch", "G LR", "D LR", "D Total", "D Real", "D Fake",
        "G Total", "G Adv", "G L1", "Val L1 Error", "Val SSIM",
        "Val PSNR (dB)", "Best Checkpoint?", "Observations"
    ]
    for col_idx, h in enumerate(train_headers, start=1):
        c = ws_train.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = crimson_fill
        c.alignment = center_align

    # =========================================================================
    # SHEET 3: Style_Evaluation
    # =========================================================================
    ws_style = wb.create_sheet(title="Style_Evaluation")
    ws_style.views.sheetView[0].showGridLines = True
    ws_style.merge_cells("A1:J1")
    ws_style["A1"] = "Per-Style Validation Performance Breakdown"
    ws_style["A1"].font = title_font

    style_headers = [
        "Evaluation Stage / Epoch", "Style Category", "Val Samples",
        "Mean L1 Error", "Mean SSIM", "Mean PSNR (dB)", "Reconstruction Quality", "Notes"
    ]
    for col_idx, h in enumerate(style_headers, start=1):
        c = ws_style.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = dark_teal_fill
        c.alignment = center_align

    # =========================================================================
    # SHEET 4: Optuna_Study
    # =========================================================================
    ws_opt = wb.create_sheet(title="Optuna_Study")
    ws_opt.views.sheetView[0].showGridLines = True
    ws_opt.merge_cells("A1:M1")
    ws_opt["A1"] = "Task 4 Optuna Hyperparameter Optimization Study"
    ws_opt["A1"].font = title_font

    opt_headers = [
        "Trial", "G LR", "D LR", "Batch Size", "Base Channels", "Dropout",
        "Embed Dim", "Lambda L1", "Val SSIM", "Val PSNR (dB)", "Val L1", "Pruned?", "State"
    ]
    for col_idx, h in enumerate(opt_headers, start=1):
        c = ws_opt.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = slate_fill
        c.alignment = center_align

    # Auto-adjust column widths
    for ws in [ws_ov, ws_train, ws_style, ws_opt]:
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
    print(f"[ExcelTracker] Initialized Task 4 workbook at: {output_path}")
    return wb


def append_cgan_epoch(
    epoch: int,
    g_lr: float,
    d_lr: float,
    d_metrics: Dict[str, float],
    g_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_best: bool = False,
    notes: str = "",
    excel_path: str = EXCEL_PATH,
):
    """Log an epoch of cGAN training."""
    if not os.path.exists(excel_path):
        create_styled_task4_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Training_Progress"]
    next_row = ws.max_row + 1

    green_fill = PatternFill(start_color="D4EFDF", end_color="D4EFDF", fill_type="solid")

    row_data = [
        epoch,
        f"{g_lr:.6f}",
        f"{d_lr:.6f}",
        round(d_metrics.get("d_total", 0.0), 4),
        round(d_metrics.get("d_real", 0.0), 4),
        round(d_metrics.get("d_fake", 0.0), 4),
        round(g_metrics.get("g_total", 0.0), 4),
        round(g_metrics.get("g_adv", 0.0), 4),
        round(g_metrics.get("g_l1", 0.0), 4),
        round(val_metrics.get("val_l1", 0.0), 4),
        round(val_metrics.get("val_ssim", 0.0), 4),
        round(val_metrics.get("val_psnr", 0.0), 2),
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


def append_style_eval(
    stage_name: str,
    per_style_results: Dict[str, Dict[str, float]],
    excel_path: str = EXCEL_PATH,
):
    """Log per-style breakdown."""
    if not os.path.exists(excel_path):
        create_styled_task4_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Style_Evaluation"]

    for s_name, res in per_style_results.items():
        row_data = [
            stage_name,
            s_name,
            res.get("count", 0),
            round(res.get("l1", 0.0), 4),
            round(res.get("ssim", 0.0), 4),
            round(res.get("psnr", 0.0), 2),
            "Optimal" if res.get("ssim", 0.0) > 0.70 else "Developing",
            f"Evaluated on {res.get('count', 0)} pairs",
        ]
        next_row = ws.max_row + 1
        for c_idx, val in enumerate(row_data, start=1):
            cell = ws.cell(row=next_row, column=c_idx, value=val)
            cell.font = Font(name="Segoe UI", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(excel_path)


def append_optuna_trial(
    trial_num: int,
    params: Dict[str, Any],
    val_ssim: float,
    val_psnr: float,
    val_l1: float,
    pruned: bool,
    excel_path: str = EXCEL_PATH,
):
    """Log an Optuna trial result."""
    if not os.path.exists(excel_path):
        create_styled_task4_tracker(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Optuna_Study"]
    next_row = ws.max_row + 1

    row_data = [
        trial_num,
        f"{params.get('g_lr', 0.0):.6f}",
        f"{params.get('d_lr', 0.0):.6f}",
        params.get("batch_size", 16),
        params.get("base_channels", 64),
        round(params.get("dropout", 0.0), 3),
        params.get("embed_dim", 16),
        round(params.get("lambda_l1", 100.0), 1),
        round(val_ssim, 4) if val_ssim is not None else "N/A",
        round(val_psnr, 2) if val_psnr is not None else "N/A",
        round(val_l1, 4) if val_l1 is not None else "N/A",
        "YES" if pruned else "No",
        "Pruned" if pruned else "Completed",
    ]
    for c_idx, val in enumerate(row_data, start=1):
        cell = ws.cell(row=next_row, column=c_idx, value=val)
        cell.font = Font(name="Segoe UI", size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(excel_path)


if __name__ == "__main__":
    create_styled_task4_tracker()
