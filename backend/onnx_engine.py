"""
ONNX Runtime Engine for Task 1: Universal Restoration
Loads and runs inference with millisecond latency measurement.
"""

import io
import time
import base64
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.cm as cm


class UniversalRestorationEngine:
    def __init__(self, onnx_model_path: str):
        self.model_path = onnx_model_path
        # Start session on CPU (or CUDA if available)
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"UniversalRestorationEngine initialized from {onnx_model_path} with {providers[0]}")

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        """Converts PIL image (128x128 RGB) to float32 tensor of shape (1, 3, 128, 128) in [0, 1]."""
        resized = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0  # (128, 128, 3)
        tensor = np.transpose(arr, (2, 0, 1))               # (3, 128, 128)
        tensor = np.expand_dims(tensor, axis=0)             # (1, 3, 128, 128)
        return tensor

    def postprocess(self, tensor: np.ndarray) -> Image.Image:
        """Converts float32 tensor (1, 3, 128, 128) to PIL Image."""
        arr = tensor[0]                                     # (3, 128, 128)
        arr = np.transpose(arr, (1, 2, 0))                  # (128, 128, 3)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def compute_error_map(self, clean_pil: Image.Image, restored_pil: Image.Image) -> Image.Image:
        """Generates heatmap of absolute error |restored - clean|."""
        c = np.array(clean_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        r = np.array(restored_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        diff = np.mean(np.abs(r - c), axis=-1)  # (128, 128)
        # Normalize and apply magma colormap
        norm_diff = np.clip(diff / 0.35, 0.0, 1.0)
        heatmap = cm.magma(norm_diff)[:, :, :3]  # drop alpha
        heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
        return Image.fromarray(heatmap_uint8)

    def to_base64(self, pil_img: Image.Image) -> str:
        """Encodes PIL image as JPEG base64 data URI."""
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def run_restoration(
        self,
        corrupted_pil: Image.Image,
        clean_pil: Image.Image = None,
    ) -> dict:
        """
        Runs full ONNX inference and returns base64 images and timing info.
        """
        input_tensor = self.preprocess(corrupted_pil)

        # Timed inference
        start_time = time.perf_counter()
        onnx_outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        restored_tensor = onnx_outputs[0]
        restored_pil = self.postprocess(restored_tensor)

        result = {
            "latency_ms": round(latency_ms, 2),
            "input_image": self.to_base64(corrupted_pil),
            "restored_image": self.to_base64(restored_pil),
            "error_map_image": None,
            "metrics": {},
        }

        if clean_pil is not None:
            clean_128 = clean_pil.convert("RGB").resize((128, 128))
            result["clean_image"] = self.to_base64(clean_128)
            error_map_pil = self.compute_error_map(clean_128, restored_pil)
            result["error_map_image"] = self.to_base64(error_map_pil)

            # Quick numpy metric calculations
            c = np.array(clean_128, dtype=np.float32) / 255.0
            r = np.array(restored_pil, dtype=np.float32) / 255.0
            l1 = float(np.mean(np.abs(r - c)))
            mse = float(np.mean((r - c) ** 2))
            psnr = round(10 * np.log10(1.0 / max(1e-10, mse)), 2)
            result["metrics"] = {"l1_error": round(l1, 4), "psnr_db": psnr}

        return result
