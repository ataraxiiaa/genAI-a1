"""
ONNX Export and Consistency Verification for Task 2: Specialist Autoencoders
Exports trained specialist autoencoders to ONNX format:
- checkpoints/specialist_noise.onnx
- checkpoints/specialist_blur.onnx
- checkpoints/specialist_occlusion.onnx

Verifies numerical consistency between PyTorch and ONNX Runtime (max difference < 1e-4).
"""

import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)

from specialists import get_specialist


def export_single_specialist(specialist_type: str, checkpoint_path: str, output_onnx_path: str, device: str = "cpu"):
    print("\n" + "=" * 70)
    print(f"EXPORTING SPECIALIST TO ONNX: {specialist_type.upper()}")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load PyTorch Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    base_channels = config.get("base_channels", 64)
    latent_dim = config.get("latent_dim", 256)

    print(f"Loaded checkpoint from Epoch {ckpt.get('epoch', '?')} (Val Loss: {ckpt.get('val_loss', 0.0):.4f}, SSIM: {ckpt.get('val_ssim', 0.0):.4f})")
    print(f"Config: base_channels={base_channels}, latent_dim={latent_dim}")

    model = get_specialist(specialist_type, base_channels=base_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("PyTorch specialist loaded and switched to eval() mode.")

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    input_names = ["corrupted_image"]
    output_names = ["restored_image"]
    dynamic_axes = {
        "corrupted_image": {0: "batch_size"},
        "restored_image": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"ONNX model saved to: {output_onnx_path} ({file_size_mb:.2f} MB)")

    # 3. Verify PyTorch vs ONNX Runtime Consistency
    with torch.no_grad():
        pytorch_out = model(dummy_input).cpu().numpy()

    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    abs_diff = np.abs(pytorch_out - ort_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"PyTorch Output Shape: {pytorch_out.shape}")
    print(f"ONNX Output Shape:    {ort_out.shape}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    print(f"Max Absolute Difference:  {max_diff:.8f}")

    if max_diff < 1e-4:
        print(f"Verification PASSED: {specialist_type} ONNX Runtime output matches PyTorch (max diff < 1e-4)!")
    else:
        print(f"Warning: Discrepancy exceeded 1e-4 threshold.")


def main():
    parser = argparse.ArgumentParser(description="Export Trained Specialists to ONNX")
    parser.add_argument(
        "--specialist",
        type=str,
        default="all",
        choices=["noise", "blur", "occlusion", "all"],
        help="Specialist to export ('noise', 'blur', 'occlusion', or 'all')",
    )
    args = parser.parse_args()

    targets = ["noise", "blur", "occlusion"] if args.specialist == "all" else [args.specialist]
    checkpoints_dir = os.path.join(PARENT_DIR, "checkpoints")

    for target in targets:
        ckpt_path = os.path.join(checkpoints_dir, f"best_specialist_{target}.pt")
        onnx_path = os.path.join(checkpoints_dir, f"specialist_{target}.onnx")
        if os.path.exists(ckpt_path):
            export_single_specialist(target, ckpt_path, onnx_path)
        else:
            print(f"Skipping {target}: checkpoint not found at {ckpt_path}")


if __name__ == "__main__":
    main()
