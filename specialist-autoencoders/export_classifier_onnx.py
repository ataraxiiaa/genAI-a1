"""
ONNX Export and Consistency Verification for Task 2: Corruption Classifier
Exports trained CorruptionClassifier to checkpoints/corruption_classifier.onnx
and verifies output parity between PyTorch and ONNX Runtime.
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

from classifier import CorruptionClassifier, CLASS_NAMES, NUM_CLASSES


def export_classifier_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    device: str = "cpu",
):
    print("=" * 70)
    print("TASK 2: EXPORTING CORRUPTION CLASSIFIER TO ONNX")
    print("=" * 70)

    # 1. Load PyTorch Model
    print(f"\n--- Step 1: Loading Trained PyTorch Checkpoint ---")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    base_channels = config.get("base_channels", 32)
    dropout = config.get("dropout", 0.1)

    print(f"Loaded checkpoint from Epoch {ckpt.get('epoch', '?')} (Val Acc: {ckpt.get('val_acc', 0.0) * 100:.2f}%)")
    print(f"Model Configuration: base_channels={base_channels}, dropout={dropout}")

    model = CorruptionClassifier(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
        dropout_rate=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("PyTorch model successfully loaded and set to eval() mode.")

    # 2. Export to ONNX
    print(f"\n--- Step 2: Tracing and Exporting to ONNX ---")
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    input_names = ["input_image"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "logits": {0: "batch_size"},
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
    print(f"ONNX model exported to: {output_onnx_path}")
    print(f"Exported model size: {file_size_mb:.2f} MB")

    # 3. Verify PyTorch vs ONNX Runtime Consistency
    print(f"\n--- Step 3: Verifying Consistency (PyTorch vs ONNX Runtime) ---")
    with torch.no_grad():
        pytorch_logits = model(dummy_input).cpu().numpy()
        pytorch_probs = torch.softmax(torch.from_numpy(pytorch_logits), dim=-1).numpy()

    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_logits = ort_session.run(None, ort_inputs)[0]
    ort_probs = np.exp(ort_logits) / np.sum(np.exp(ort_logits), axis=-1, keepdims=True)

    abs_diff = np.abs(pytorch_logits - ort_logits)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"PyTorch Output Shape: {pytorch_logits.shape}")
    print(f"ONNX Output Shape:    {ort_logits.shape}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    print(f"Max Absolute Difference:  {max_diff:.8f}")

    if max_diff < 1e-4:
        print("Verification PASSED: ONNX Runtime matches PyTorch within numerical tolerance (1e-4)!")
    else:
        print("Warning: Numerical discrepancy exceeded 1e-4 threshold.")

    print("\nSample Probabilities Comparison:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Class [{name}]: PyTorch={pytorch_probs[0][i]:.4f} | ONNX={ort_probs[0][i]:.4f}")

    print("\n" + "=" * 70)
    print("CLASSIFIER ONNX EXPORT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Trained Classifier to ONNX")
    default_ckpt = os.path.join(CURRENT_DIR, "checkpoints", "best_corruption_classifier.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(PARENT_DIR, "checkpoints", "best_corruption_classifier.pt")

    default_onnx = os.path.join(PARENT_DIR, "checkpoints", "corruption_classifier.onnx")

    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default=default_onnx, help="Path to save output ONNX model")
    args = parser.parse_args()

    export_classifier_to_onnx(args.checkpoint, args.output)
