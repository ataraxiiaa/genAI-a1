"""
Task 4: Conditional GAN Generator ONNX Export & Parity Verification
Exports trained UNetGenerator to checkpoints/cgan_generator.onnx with dynamic batch axes
and validates numerical parity with ONNX Runtime.

Input Specifications:
- 'photo': Tensor (B, 3, 128, 128), float32 in [-1.0, 1.0]
- 'style_idx': Tensor (B,), int64 in {0, 1, 2}
Output Specifications:
- 'sketch': Tensor (B, 3, 128, 128), float32 in [-1.0, 1.0]
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import UNetGenerator


def export_generator_onnx(
    checkpoint_path: str = "checkpoints/best_cgan_generator.pt",
    output_path: str = "checkpoints/cgan_generator.onnx",
    config_path: str = "checkpoints/best_cgan_hyperparameters.json",
    base_channels: int = 64,
    embed_dim: int = 16,
    dropout: float = 0.0,
    opset_version: int = 17,
    verify: bool = True,
) -> str:
    print("=" * 75)
    print("TASK 4: EXPORTING CGAN GENERATOR TO ONNX")
    print("=" * 75)

    # 1. Load config if present to match model architecture
    if os.path.exists(config_path):
        print(f"Reading architecture hyperparams from config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("best_params", cfg)
            base_channels = params.get("base_channels", base_channels)
            embed_dim = params.get("embed_dim", embed_dim)
            dropout = params.get("dropout", dropout)
            print(f"  base_channels={base_channels}, embed_dim={embed_dim}, dropout={dropout}")

    # 2. Build PyTorch Generator
    device = torch.device("cpu")
    model = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_styles=3,
        dropout=dropout,
    ).to(device)

    # 3. Load Checkpoint Weights if available
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
            state_dict = ckpt["generator_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and any("down1" in k for k in ckpt.keys()):
            state_dict = ckpt
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded trained generator weights from: {checkpoint_path}")
    else:
        print(f"Notice: Checkpoint '{checkpoint_path}' not found. Exporting initialized architecture.")

    model.eval()

    # 4. Define Dummy Inputs
    dummy_photo = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    dummy_style = torch.tensor([0], dtype=torch.long)

    # 5. Export to ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_names = ["photo", "style_idx"]
    output_names = ["sketch"]
    dynamic_axes = {
        "photo": {0: "batch_size"},
        "style_idx": {0: "batch_size"},
        "sketch": {0: "batch_size"},
    }

    print(f"Exporting to ONNX at: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_photo, dummy_style),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 6. Verify ONNX Model Graph
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX export successful! Model file size: {file_size_mb:.2f} MB")

    # 7. Numerical Parity Verification with ONNX Runtime
    if verify:
        print("-" * 75)
        print("RUNNING NUMERICAL PARITY VERIFICATION (PyTorch vs ONNX Runtime)")
        print("-" * 75)
        ort_sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        # Test Case 1: Batch size 1, Style 1
        photo_b1 = np.random.uniform(-1.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
        style_b1 = np.array([1], dtype=np.int64)

        with torch.no_grad():
            pt_out_b1 = model(torch.from_numpy(photo_b1), torch.from_numpy(style_b1)).numpy()
        ort_out_b1 = ort_sess.run(None, {"photo": photo_b1, "style_idx": style_b1})[0]

        diff_b1 = np.max(np.abs(pt_out_b1 - ort_out_b1))
        print(f"Test 1 (B=1, Style 1): Max Absolute Difference = {diff_b1:.8f}")
        assert diff_b1 < 1e-4, f"Parity mismatch in Test 1: max diff {diff_b1} >= 1e-4"

        # Test Case 2: Dynamic Batch Size 3, All 3 Styles simultaneously
        photo_b3 = np.random.uniform(-1.0, 1.0, (3, 3, 128, 128)).astype(np.float32)
        style_b3 = np.array([0, 1, 2], dtype=np.int64)

        with torch.no_grad():
            pt_out_b3 = model(torch.from_numpy(photo_b3), torch.from_numpy(style_b3)).numpy()
        ort_out_b3 = ort_sess.run(None, {"photo": photo_b3, "style_idx": style_b3})[0]

        diff_b3 = np.max(np.abs(pt_out_b3 - ort_out_b3))
        print(f"Test 2 (B=3, Styles [0, 1, 2]): Max Absolute Difference = {diff_b3:.8f}")
        assert diff_b3 < 1e-4, f"Parity mismatch in Test 2: max diff {diff_b3} >= 1e-4"

        print("=" * 75)
        print("VERIFICATION COMPLETE: PyTorch and ONNX Runtime outputs match exactly (< 1e-4 diff)!")
        print(f"Verified Dynamic Batch axis across single and multi-image inference.")
        print("=" * 75)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Task 4 Face-to-Sketch UNetGenerator to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_cgan_generator.pt", help="Path to trained PyTorch weights")
    parser.add_argument("--output", type=str, default="checkpoints/cgan_generator.onnx", help="Target ONNX export path")
    parser.add_argument("--config", type=str, default="checkpoints/best_cgan_hyperparameters.json", help="Path to hyperparameter JSON")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count")
    parser.add_argument("--embed_dim", type=int, default=16, help="Style embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no_verify", action="store_true", help="Skip numerical parity check")

    args = parser.parse_args()

    export_generator_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
