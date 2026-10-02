"""
Task 3: ONNX Export and Numerical Parity Verification for Soft MoE Pipeline
Export Target:
- checkpoints/soft_moe.onnx

Pipeline Specifications:
- Input:  'corrupted_image' of shape (B, 3, 128, 128), float32 in range [0, 1]
- Outputs:
    1. 'restored_image': (B, 3, 128, 128), float32 in range [0, 1]
    2. 'routing_weights': (B, 4), float32 in range [0, 1] [clean, noise, blur, occlusion]
- Dynamic axes on batch size B for real-time deployment
- Verification against ONNX Runtime to guarantee absolute difference < 1e-4
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECIALIST_DIR = os.path.join(WORKSPACE_ROOT, "specialist-autoencoders")
CURRENT_DIR = os.path.dirname(__file__)

for p in [WORKSPACE_ROOT, SPECIALIST_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from soft_moe import SoftMoERestoration


class SoftMoEOnnxWrapper(nn.Module):
    """
    ONNX-exportable wrapper for SoftMoERestoration.
    Returns: (restored_image, routing_weights).
    """

    def __init__(self, moe_model: SoftMoERestoration):
        super().__init__()
        self.moe = moe_model

    def forward(self, x: torch.Tensor):
        recon, weights, _ = self.moe(x)
        return recon, weights


def export_soft_moe_onnx(
    checkpoint_path: str = "checkpoints/best_soft_moe.pt",
    output_path: str = "checkpoints/soft_moe.onnx",
    opset_version: int = 17,
):
    print("=" * 70)
    print("Exporting Differentiable Soft MoE Pipeline to ONNX")
    print("=" * 70)

    device = torch.device("cpu")
    model = SoftMoERestoration().to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded trained checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Loading Task 2 specialist weights.")
        model.load_pretrained_weights(
            classifier_path="checkpoints/best_corruption_classifier.pt",
            noise_path="checkpoints/best_specialist_noise.pt",
            blur_path="checkpoints/best_specialist_blur.pt",
            occlusion_path="checkpoints/best_specialist_occlusion.pt",
            device=device,
        )

    model.eval()
    wrapper = SoftMoEOnnxWrapper(model).to(device)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_names = ["corrupted_image"]
    output_names = ["restored_image", "routing_weights"]
    dynamic_axes = {
        "corrupted_image": {0: "batch_size"},
        "restored_image": {0: "batch_size"},
        "routing_weights": {0: "batch_size"},
    }

    print(f"Exporting ONNX model to: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported successfully! File size: {file_size_mb:.2f} MB")

    # 1. Check with ONNX library
    print("Validating ONNX graph integrity...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  --> ONNX graph check PASSED!")

    # 2. Numerical Parity with ONNX Runtime
    print("Verifying numerical parity between PyTorch and ONNX Runtime...")
    wrapper.eval()
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    test_input = torch.rand(2, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        torch_recon, torch_weights = wrapper(test_input)

    ort_inputs = {session.get_inputs()[0].name: test_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    ort_recon, ort_weights = ort_outputs[0], ort_outputs[1]

    diff_recon = np.max(np.abs(torch_recon.numpy() - ort_recon))
    diff_weights = np.max(np.abs(torch_weights.numpy() - ort_weights))

    print(f"  Reconstruction Max Absolute Difference: {diff_recon:.2e}")
    print(f"  Routing Weights Max Absolute Difference: {diff_weights:.2e}")

    assert diff_recon < 1e-4, f"Numerical parity failure in reconstruction: {diff_recon}"
    assert diff_weights < 1e-4, f"Numerical parity failure in weights: {diff_weights}"

    print("Numerical parity check PASSED (< 1e-4)!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Export Soft MoE Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_soft_moe.pt")
    parser.add_argument("--output", type=str, default="checkpoints/soft_moe.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_soft_moe_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
