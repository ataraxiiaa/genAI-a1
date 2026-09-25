"""
ONNX Export and Verification Script for Task 1: Universal Denoising Autoencoder
Step 1: Load the trained model and switch to evaluation mode.
"""

import os
import sys
import torch
import numpy as np
import onnxruntime as ort

# Ensure current directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from AE import DenoisingAutoencoder


def load_model(checkpoint_path: str = "checkpoints/best_universal_ae.pt"):
    print("--- Step 1: Loading Trained PyTorch Model ---")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. Load checkpoint dictionary on CPU
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    epoch = checkpoint.get("epoch", "unknown")
    best_loss = checkpoint.get("val_loss", "unknown")

    print(f"Checkpoint found from Epoch {epoch} with Best Val Loss: {best_loss:.4f}")

    # 2. Extract architecture hyperparameters
    base_channels = config.get("base_channels", 64)
    latent_dim = config.get("latent_dim", 256)
    dropout = config.get("dropout", 0.05)

    print(f"Model Configuration: base_channels={base_channels}, latent_dim={latent_dim}, dropout={dropout}")

    # 3. Instantiate the architecture
    model = DenoisingAutoencoder(
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout_rate=dropout,
    )

    # 4. Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # 5. Set to evaluation mode (Turns off Dropout & locks BatchNorm)
    model.eval()

    print("Model successfully loaded and switched to eval() mode.\n")
    return model


def export_onnx(model: torch.nn.Module, onnx_path: str = "checkpoints/universal_ae.onnx") -> torch.Tensor:
    print("--- Step 2: Tracing and Exporting to ONNX ---")
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    # 1. Create a dummy input tensor matching image dimensions (Batch=1, Channels=3, 128, 128)
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=False)
    print(f"Created dummy input tensor with shape: {list(dummy_input.shape)}")

    # 2. Call torch.onnx.export
    print(f"Exporting model to: {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,             # Store the trained parameter weights inside the file
        opset_version=17,               # Standard modern ONNX operator set
        do_constant_folding=True,       # Optimizes constant operations for faster speed
        input_names=["input"],          # Clean name for input layer
        output_names=["output"],        # Clean name for output layer
        dynamic_axes={                  # Allows variable batch sizes (e.g. 1 image or a batch of 8)
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model successfully exported! File size: {file_size_mb:.2f} MB\n")
    return dummy_input


def verify_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str = "checkpoints/universal_ae.onnx",
):
    print("--- Step 3: Verifying Consistency (PyTorch vs ONNX Runtime) ---")

    # 1. Run inference using PyTorch
    with torch.no_grad():
        pytorch_out = model(dummy_input).cpu().numpy()

    # 2. Run inference using ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_inputs = {input_name: dummy_input.cpu().numpy()}
    onnx_out = session.run([output_name], onnx_inputs)[0]

    # 3. Calculate difference between PyTorch and ONNX outputs
    mean_diff = float(np.mean(np.abs(pytorch_out - onnx_out)))
    max_diff = float(np.max(np.abs(pytorch_out - onnx_out)))

    print(f"PyTorch Output Shape: {pytorch_out.shape}")
    print(f"ONNX Output Shape:    {onnx_out.shape}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    print(f"Max Absolute Difference:  {max_diff:.8f}")

    # 4. Strict assertion: difference must be < 1e-4
    tolerance = 1e-4
    assert max_diff < tolerance, f"Verification failed! Max difference {max_diff} exceeds tolerance {tolerance}"
    print(f"Verification PASSED: ONNX Runtime output matches PyTorch (max difference < {tolerance})!\n")


if __name__ == "__main__":
    model = load_model()
    dummy_input = export_onnx(model)
    verify_onnx(model, dummy_input)


