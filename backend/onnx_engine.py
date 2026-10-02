"""
ONNX Runtime Engine for Task 1: Universal Restoration
Loads and runs inference with millisecond latency measurement.
"""

import os
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


class HardRoutingEngine:
    """
    ONNX Runtime Engine for Task 2: Hard-Routed Restoration.
    Coordinates the 4-Class Corruption Classifier and 3 Specialist Autoencoders + Identity Bypass.
    """
    CLASS_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
    SHORT_NAMES = ["clean", "noise", "blur", "occlusion"]
    EXPERT_NAMES = {
        0: "Clean Identity Bypass (Zero Compute)",
        1: "Noise Specialist Autoencoder (Salt & Pepper)",
        2: "Blur Specialist Autoencoder (Gaussian Blur)",
        3: "Occlusion Specialist Autoencoder (Inpainting)",
    }

    def __init__(self, checkpoints_dirs: list):
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.providers = providers

        self.classifier_session = None
        self.specialist_sessions = {}

        # Search candidate directories for model files
        for cdir in checkpoints_dirs:
            if not os.path.exists(cdir):
                continue

            cls_p = os.path.join(cdir, "corruption_classifier.onnx")
            if self.classifier_session is None and os.path.exists(cls_p):
                try:
                    self.classifier_session = ort.InferenceSession(cls_p, providers=providers)
                    print(f"HardRoutingEngine: Loaded Classifier from {cls_p}")
                except Exception as e:
                    print(f"HardRoutingEngine: Failed to load classifier: {e}")

            for s_type, s_id in [("noise", 1), ("blur", 2), ("occlusion", 3)]:
                sp_p = os.path.join(cdir, f"specialist_{s_type}.onnx")
                if s_id not in self.specialist_sessions and os.path.exists(sp_p):
                    try:
                        self.specialist_sessions[s_id] = ort.InferenceSession(sp_p, providers=providers)
                        print(f"HardRoutingEngine: Loaded {s_type.upper()} Specialist from {sp_p}")
                    except Exception as e:
                        print(f"HardRoutingEngine: Failed to load {s_type} specialist: {e}")

    def is_ready(self) -> bool:
        return self.classifier_session is not None and len(self.specialist_sessions) >= 3

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        resized = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def postprocess(self, tensor: np.ndarray) -> Image.Image:
        arr = tensor[0]
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def to_base64(self, pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def compute_error_map(self, clean_pil: Image.Image, restored_pil: Image.Image) -> Image.Image:
        c = np.array(clean_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        r = np.array(restored_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        diff = np.mean(np.abs(r - c), axis=-1)
        norm_diff = np.clip(diff / 0.35, 0.0, 1.0)
        heatmap = cm.magma(norm_diff)[:, :, :3]
        return Image.fromarray((heatmap * 255.0).astype(np.uint8))

    def run_hard_routing(
        self,
        corrupted_pil: Image.Image,
        clean_pil: Image.Image = None,
        mode: str = "predicted",
        oracle_label: int = None,
    ) -> dict:
        start_time = time.perf_counter()
        inp_tensor = self.preprocess(corrupted_pil)

        # 1. Run Classifier
        probs_dict = {name: 0.25 for name in self.CLASS_NAMES}
        pred_idx = 0

        if self.classifier_session is not None:
            input_name = self.classifier_session.get_inputs()[0].name
            logits = self.classifier_session.run(None, {input_name: inp_tensor})[0]
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            pred_idx = int(np.argmax(probs[0]))
            probs_dict = {name: round(float(probs[0][i]), 4) for i, name in enumerate(self.CLASS_NAMES)}

        routed_idx = oracle_label if (mode == "oracle" and oracle_label is not None) else pred_idx

        # 2. Dispatch to Expert
        if routed_idx == 0:
            # Clean Identity Bypass
            restored_pil = corrupted_pil.convert("RGB").resize((128, 128))
        elif routed_idx in self.specialist_sessions:
            sess = self.specialist_sessions[routed_idx]
            inp_name = sess.get_inputs()[0].name
            out_tensor = sess.run(None, {inp_name: inp_tensor})[0]
            restored_pil = self.postprocess(out_tensor)
        else:
            restored_pil = corrupted_pil.convert("RGB").resize((128, 128))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = {
            "mode": mode,
            "predicted_class_id": pred_idx,
            "predicted_corruption": self.CLASS_NAMES[pred_idx],
            "routed_class_id": routed_idx,
            "selected_expert": self.EXPERT_NAMES.get(routed_idx, "Unknown Expert"),
            "probabilities": probs_dict,
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

            c = np.array(clean_128, dtype=np.float32) / 255.0
            r = np.array(restored_pil, dtype=np.float32) / 255.0
            l1 = float(np.mean(np.abs(r - c)))
            mse = float(np.mean((r - c) ** 2))
            psnr = round(10 * np.log10(1.0 / max(1e-10, mse)), 2)
            result["metrics"] = {"l1_error": round(l1, 4), "psnr_db": psnr}

        return result


class SoftMoEEngine:
    """
    ONNX Runtime Engine for Task 3: Differentiable Soft Mixture-of-Experts.
    Restores degraded images using continuous soft routing across 4 branches.
    """
    BRANCH_NAMES = ["Clean Bypass", "Noise Specialist", "Blur Specialist", "Occlusion Specialist"]
    BRANCH_KEYS = ["clean_identity", "salt_and_pepper_expert", "gaussian_blur_expert", "occlusion_expert"]

    def __init__(self, onnx_model_path: str):
        self.model_path = onnx_model_path
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.providers = providers
        print(f"SoftMoEEngine: Initialized Soft MoE from {onnx_model_path} with {providers[0]}")

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        resized = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def postprocess(self, tensor: np.ndarray) -> Image.Image:
        arr = tensor[0]
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def compute_error_map(self, clean_pil: Image.Image, restored_pil: Image.Image) -> Image.Image:
        c = np.array(clean_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        r = np.array(restored_pil.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        diff = np.mean(np.abs(r - c), axis=-1)
        norm_diff = np.clip(diff / 0.35, 0.0, 1.0)
        heatmap = cm.magma(norm_diff)[:, :, :3]
        return Image.fromarray((heatmap * 255.0).astype(np.uint8))

    def to_base64(self, pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def run_restoration(self, corrupted_pil: Image.Image, clean_pil: Image.Image = None) -> dict:
        inp_tensor = self.preprocess(corrupted_pil)
        start_time = time.perf_counter()
        outputs = self.session.run(None, {"corrupted_image": inp_tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        restored_tensor = outputs[0]
        routing_weights = outputs[1][0]  # shape (4,)

        restored_pil = self.postprocess(restored_tensor)

        weights_dict = {
            self.BRANCH_KEYS[i]: round(float(routing_weights[i]), 4)
            for i in range(4)
        }
        dominant_idx = int(np.argmax(routing_weights))

        result = {
            "latency_ms": round(latency_ms, 2),
            "input_image": self.to_base64(corrupted_pil.convert("RGB").resize((128, 128))),
            "restored_image": self.to_base64(restored_pil),
            "routing_weights": weights_dict,
            "dominant_expert": self.BRANCH_NAMES[dominant_idx],
            "dominant_expert_key": self.BRANCH_KEYS[dominant_idx],
            "error_map_image": None,
            "metrics": {},
        }

        if clean_pil is not None:
            clean_128 = clean_pil.convert("RGB").resize((128, 128))
            result["clean_image"] = self.to_base64(clean_128)
            error_map_pil = self.compute_error_map(clean_128, restored_pil)
            result["error_map_image"] = self.to_base64(error_map_pil)

            c = np.array(clean_128, dtype=np.float32) / 255.0
            r = np.array(restored_pil, dtype=np.float32) / 255.0
            l1 = float(np.mean(np.abs(r - c)))
            mse = float(np.mean((r - c) ** 2))
            psnr = round(10 * np.log10(1.0 / max(1e-10, mse)), 2)
            result["metrics"] = {"l1_error": round(l1, 4), "psnr_db": psnr}

        return result


class FaceToSketchEngine:
    """
    ONNX Runtime Engine for Task 4: Style-Conditioned Face-to-Sketch Generator (Pix2Pix cGAN).
    Accepts facial photograph, generates high-fidelity sketch conditioned on Style 1, 2, or 3.
    """
    STYLE_CONFIGS = {
        0: {
            "name": "Style 1: Shaded Pencil",
            "short_name": "Style 1",
            "desc": "Gradated pencil shading, natural graphite textures, soft dimensional depth",
            "badge": "Shaded Pencil",
        },
        1: {
            "name": "Style 2: Clean Contour",
            "short_name": "Style 2",
            "desc": "Minimalist clean outlines, crisp structural contours, delicate line art",
            "badge": "Clean Contour",
        },
        2: {
            "name": "Style 3: Graphic Comic",
            "short_name": "Style 3",
            "desc": "High-contrast ink outlines, stylized expressive features, comic illustration",
            "badge": "Graphic Comic",
        },
    }

    def __init__(self, onnx_model_path: str):
        self.model_path = onnx_model_path
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.providers = providers
        print(f"FaceToSketchEngine: Initialized cGAN generator from {onnx_model_path} with {providers[0]}")

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        """Resizes to 128x128 RGB and normalizes to [-1.0, 1.0]."""
        resized = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 127.5 - 1.0  # [-1.0, 1.0]
        tensor = np.transpose(arr, (2, 0, 1))                  # (3, 128, 128)
        return np.expand_dims(tensor, axis=0)                  # (1, 3, 128, 128)

    def postprocess(self, tensor: np.ndarray) -> Image.Image:
        """Denormalizes [-1.0, 1.0] to [0, 255] uint8 PIL Image."""
        arr = tensor[0]                                        # (3, 128, 128)
        arr = np.transpose(arr, (1, 2, 0))                     # (128, 128, 3)
        denorm = ((arr * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(denorm)

    def to_base64(self, pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def run_generation(self, photo_pil: Image.Image, style_idx: int = 0) -> dict:
        """
        Runs style-conditioned sketch generation on the input photograph.
        """
        # Ensure style_idx in {0, 1, 2}
        style_idx = max(0, min(2, int(style_idx)))
        inp_photo = self.preprocess(photo_pil)
        inp_style = np.array([style_idx], dtype=np.int64)

        start_time = time.perf_counter()
        sketch_tensor = self.session.run(None, {"photo": inp_photo, "style_idx": inp_style})[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        sketch_pil = self.postprocess(sketch_tensor)
        photo_128 = photo_pil.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)

        cfg = self.STYLE_CONFIGS.get(style_idx, self.STYLE_CONFIGS[0])

        return {
            "style_idx": style_idx,
            "style_name": cfg["name"],
            "style_short": cfg["short_name"],
            "style_description": cfg["desc"],
            "style_badge": cfg["badge"],
            "latency_ms": round(latency_ms, 2),
            "input_photo": self.to_base64(photo_128),
            "sketch_image": self.to_base64(sketch_pil),
            "model_engine": "ONNX Runtime (Pix2Pix U-Net cGAN)",
            "resolution": "128x128",
        }


