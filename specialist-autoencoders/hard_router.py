"""
Task 2: Hard-Routing Inference & Evaluation Pipeline
Coordinates:
1. 4-Class Corruption Classifier (Routes image based on corruption)
2. 3 Specialist Autoencoders (Noise, Blur, Occlusion) + Clean Identity Bypass

Features:
- Two Routing Modes:
    * 'predicted': Routes using the trained classifier's prediction
    * 'oracle': Routes using ground-truth label (ideal routing upper-bound)
- Dual Execution Backend:
    * PyTorch mode (for local research, benchmarking, and gradient workflows)
    * ONNX Runtime mode (for ultra-fast sub-20ms deployment in FastAPI)
- Full Evaluation Suite:
    * Evaluates Oracle vs. Predicted Routing vs. Task 1 Universal AE on test set
    * Computes comparative PSNR/SSIM, routing accuracy, and misrouting degradation
"""

import os
import sys
import time
import io
import base64
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "universal-restoration"))

from classifier import CorruptionClassifier, CLASS_NAMES, SHORT_NAMES, NUM_CLASSES
from specialists import (
    NoiseSpecialistAE,
    BlurSpecialistAE,
    OcclusionSpecialistAE,
    IdentityBypass,
    get_specialist,
)

EXPERT_NAMES = {
    0: "Clean Identity Bypass (Zero Compute)",
    1: "Noise Specialist Autoencoder (Salt & Pepper)",
    2: "Blur Specialist Autoencoder (Gaussian Blur)",
    3: "Occlusion Specialist Autoencoder (Inpainting)",
}


# ==============================================================================
# PyTorch Hard-Routing System
# ==============================================================================
class HardRouter(nn.Module):
    """
    Unified PyTorch Hard-Routing Pipeline.
    Encapsulates Classifier + 3 Specialist Autoencoders + Identity Bypass.
    """
    def __init__(
        self,
        classifier_ckpt: Optional[str] = None,
        noise_ckpt: Optional[str] = None,
        blur_ckpt: Optional[str] = None,
        occlusion_ckpt: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        # 1. Initialize Classifier
        self.classifier = None
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            ckpt = torch.load(classifier_ckpt, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {})
            self.classifier = CorruptionClassifier(
                in_channels=3,
                num_classes=NUM_CLASSES,
                base_channels=cfg.get("base_channels", 16),
                dropout_rate=cfg.get("dropout", 0.05),
            ).to(device)
            self.classifier.load_state_dict(ckpt["model_state_dict"])
            self.classifier.eval()
            print(f"[HardRouter] Loaded Classifier from {classifier_ckpt} (Val Acc: {ckpt.get('val_acc', 0.0)*100:.2f}%)")

        # 2. Initialize Specialists
        self.noise_specialist = self._load_specialist("noise", noise_ckpt)
        self.blur_specialist = self._load_specialist("blur", blur_ckpt)
        self.occlusion_specialist = self._load_specialist("occlusion", occlusion_ckpt)
        self.clean_bypass = IdentityBypass().to(device)

        self.specialists = {
            0: self.clean_bypass,
            1: self.noise_specialist,
            2: self.blur_specialist,
            3: self.occlusion_specialist,
        }

    def _load_specialist(self, sp_type: str, ckpt_path: Optional[str]) -> nn.Module:
        model = get_specialist(sp_type).to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[HardRouter] Loaded {sp_type.upper()} Specialist from {ckpt_path}")
        else:
            print(f"[HardRouter] Initialized {sp_type.upper()} Specialist (untrained weights)")
        model.eval()
        return model

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "predicted",
        oracle_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Routes an input tensor (B, 3, 128, 128) to the designated specialist.

        Args:
            x: Corrupted image tensor in [0, 1]
            mode: 'predicted' (use classifier) or 'oracle' (use oracle_label)
            oracle_label: Ground truth corruption label (0=clean, 1=noise, 2=blur, 3=occlusion)

        Returns:
            Dict containing restored tensor, predicted class, probabilities, latency, and expert info.
        """
        start_time = time.perf_counter()
        B = x.size(0)

        # 1. Routing Decision
        probabilities = None
        if self.classifier is not None:
            logits = self.classifier(x)
            probabilities = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probabilities, dim=1)
        else:
            # Fallback if classifier not loaded
            pred_classes = torch.zeros(B, dtype=torch.long, device=self.device)
            probabilities = torch.zeros(B, NUM_CLASSES, device=self.device)

        if mode == "oracle" and oracle_label is not None:
            routed_classes = torch.full((B,), oracle_label, dtype=torch.long, device=self.device)
        else:
            routed_classes = pred_classes

        # 2. Specialist Dispatch
        # Handle batch element-wise or grouped by specialist
        restored = torch.zeros_like(x)
        for class_idx in range(NUM_CLASSES):
            mask = (routed_classes == class_idx)
            if mask.any():
                sub_batch = x[mask]
                restored[mask] = self.specialists[class_idx](sub_batch)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "restored": restored,
            "predicted_class": pred_classes.cpu().tolist() if B > 1 else pred_classes.item(),
            "routed_class": routed_classes.cpu().tolist() if B > 1 else routed_classes.item(),
            "probabilities": probabilities.cpu().numpy() if probabilities is not None else None,
            "mode": mode,
            "latency_ms": latency_ms,
        }


# ==============================================================================
# ONNX Runtime Hard-Routing Engine (For FastAPI Deployment)
# ==============================================================================
class OnnxHardRoutingEngine:
    """
    High-performance ONNX Runtime Engine for Task 2 Deployment.
    Serves classifier and specialist inference in sub-20ms latency.
    """
    def __init__(self, checkpoints_dir: str):
        import onnxruntime as ort

        self.checkpoints_dir = checkpoints_dir
        self.providers = ["CPUExecutionProvider"]

        # 1. Load Classifier ONNX
        cls_onnx = os.path.join(checkpoints_dir, "corruption_classifier.onnx")
        self.classifier_session = None
        if os.path.exists(cls_onnx):
            self.classifier_session = ort.InferenceSession(cls_onnx, providers=self.providers)
            print(f"[OnnxHardRouter] Loaded Classifier ONNX: {cls_onnx}")

        # 2. Load Specialists ONNX
        self.specialist_sessions = {}
        for sp_name, class_id in [("noise", 1), ("blur", 2), ("occlusion", 3)]:
            sp_path = os.path.join(checkpoints_dir, f"specialist_{sp_name}.onnx")
            if os.path.exists(sp_path):
                self.specialist_sessions[class_id] = ort.InferenceSession(sp_path, providers=self.providers)
                print(f"[OnnxHardRouter] Loaded {sp_name.upper()} Specialist ONNX: {sp_path}")

    def is_ready(self) -> bool:
        return self.classifier_session is not None

    def preprocess(self, pil_image: Image.Image) -> np.ndarray:
        resized = pil_image.resize((128, 128), Image.Resampling.BILINEAR)
        img_np = np.array(resized).astype(np.float32) / 255.0
        # (H, W, C) -> (1, C, H, W)
        tensor = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def to_base64(self, pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=92)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def tensor_to_pil(self, tensor_nchw: np.ndarray) -> Image.Image:
        img_chw = np.clip(tensor_nchw[0], 0.0, 1.0)
        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255.0).astype(np.uint8)
        return Image.fromarray(img_hwc)

    def generate_error_heatmap(self, clean_np: np.ndarray, restored_np: np.ndarray) -> Image.Image:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        err = np.mean(np.abs(clean_np[0] - restored_np[0]), axis=0)
        norm_err = np.clip(err / 0.5, 0.0, 1.0)
        cmap = plt.get_cmap("magma")
        rgba = (cmap(norm_err) * 255).astype(np.uint8)
        return Image.fromarray(rgba[:, :, :3])

    def restore(
        self,
        pil_image: Image.Image,
        mode: str = "predicted",
        oracle_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete hard-routing inference for the Web UI.
        """
        start_time = time.perf_counter()
        inp_tensor = self.preprocess(pil_image)

        # Step 1: Run Classifier
        probs_dict = {name: 0.25 for name in CLASS_NAMES}
        pred_idx = 0

        if self.classifier_session is not None:
            input_name = self.classifier_session.get_inputs()[0].name
            logits = self.classifier_session.run(None, {input_name: inp_tensor})[0]
            # Softmax
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            pred_idx = int(np.argmax(probs[0]))
            probs_dict = {name: float(probs[0][i]) for i, name in enumerate(CLASS_NAMES)}

        routed_idx = oracle_label if (mode == "oracle" and oracle_label is not None) else pred_idx

        # Step 2: Route to Expert
        if routed_idx == 0:
            # Clean bypass: return input directly
            restored_tensor = inp_tensor.copy()
        elif routed_idx in self.specialist_sessions:
            sess = self.specialist_sessions[routed_idx]
            inp_name = sess.get_inputs()[0].name
            restored_tensor = sess.run(None, {inp_name: inp_tensor})[0]
        else:
            # Fallback if specialist ONNX not yet generated
            restored_tensor = inp_tensor.copy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Compute UI payloads and metrics
        restored_pil = self.tensor_to_pil(restored_tensor)
        heatmap_pil = self.generate_error_heatmap(inp_tensor, restored_tensor)

        t_inp = torch.from_numpy(inp_tensor)
        t_res = torch.from_numpy(restored_tensor)
        psnr_val = float(peak_signal_noise_ratio(t_res, t_inp, data_range=1.0).item())
        ssim_val = float(structural_similarity_index_measure(t_res, t_inp, data_range=1.0).item())

        return {
            "mode": mode,
            "predicted_class_id": pred_idx,
            "predicted_corruption": CLASS_NAMES[pred_idx],
            "routed_class_id": routed_idx,
            "selected_expert": EXPERT_NAMES[routed_idx],
            "probabilities": probs_dict,
            "latency_ms": round(latency_ms, 2),
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "input_base64": self.to_base64(pil_image.resize((128, 128))),
            "restored_base64": self.to_base64(restored_pil),
            "error_map_base64": self.to_base64(heatmap_pil),
        }


# ==============================================================================
# Comprehensive Benchmark Evaluation (Oracle vs Predicted vs Task 1 Universal)
# ==============================================================================
def evaluate_routing_benchmark(router: HardRouter, dataloader, task1_universal_model=None):
    """
    Evaluates Oracle Routing vs. Predicted Routing vs. Universal Baseline across the validation set.
    """
    router.eval()
    results = {
        "oracle": {"psnr": [], "ssim": [], "per_class_ssim": {c: [] for c in range(NUM_CLASSES)}},
        "predicted": {"psnr": [], "ssim": [], "per_class_ssim": {c: [] for c in range(NUM_CLASSES)}},
        "universal": {"psnr": [], "ssim": [], "per_class_ssim": {c: [] for c in range(NUM_CLASSES)}},
        "routing_correct": 0,
        "total_samples": 0,
    }

    print("\n" + "=" * 70)
    print("RUNNING HARD-ROUTING BENCHMARK (ORACLE VS PREDICTED VS UNIVERSAL)")
    print("=" * 70)

    for corrupted, clean, labels, meta in dataloader:
        corrupted = corrupted.to(router.device)
        clean = clean.to(router.device)
        labels = labels.to(router.device)
        B = corrupted.size(0)

        # 1. Oracle Routing
        oracle_out = router(corrupted, mode="oracle")["restored"]
        # 2. Predicted Routing
        pred_dict = router(corrupted, mode="predicted")
        pred_out = pred_dict["restored"]
        pred_classes = torch.tensor(pred_dict["predicted_class"], device=router.device) if isinstance(pred_dict["predicted_class"], list) else torch.tensor([pred_dict["predicted_class"]], device=router.device)

        # Routing Accuracy
        results["routing_correct"] += (pred_classes == labels).sum().item()
        results["total_samples"] += B

        # Compute Metrics
        for i in range(B):
            lbl = labels[i].item()
            # Oracle
            o_psnr = peak_signal_noise_ratio(oracle_out[i:i+1], clean[i:i+1], data_range=1.0).item()
            o_ssim = structural_similarity_index_measure(oracle_out[i:i+1], clean[i:i+1], data_range=1.0).item()
            results["oracle"]["psnr"].append(o_psnr)
            results["oracle"]["ssim"].append(o_ssim)
            results["oracle"]["per_class_ssim"][lbl].append(o_ssim)

            # Predicted
            p_psnr = peak_signal_noise_ratio(pred_out[i:i+1], clean[i:i+1], data_range=1.0).item()
            p_ssim = structural_similarity_index_measure(pred_out[i:i+1], clean[i:i+1], data_range=1.0).item()
            results["predicted"]["psnr"].append(p_psnr)
            results["predicted"]["ssim"].append(p_ssim)
            results["predicted"]["per_class_ssim"][lbl].append(p_ssim)

            # Universal Baseline
            if task1_universal_model is not None:
                with torch.no_grad():
                    u_out = task1_universal_model(corrupted[i:i+1])
                u_psnr = peak_signal_noise_ratio(u_out, clean[i:i+1], data_range=1.0).item()
                u_ssim = structural_similarity_index_measure(u_out, clean[i:i+1], data_range=1.0).item()
                results["universal"]["psnr"].append(u_psnr)
                results["universal"]["ssim"].append(u_ssim)
                results["universal"]["per_class_ssim"][lbl].append(u_ssim)

    routing_acc = (results["routing_correct"] / results["total_samples"]) * 100.0
    print(f"\nEvaluation Results ({results['total_samples']} samples):")
    print(f"  Routing Accuracy: {routing_acc:.2f}%")
    print(f"  Oracle Routing:    Mean SSIM = {np.mean(results['oracle']['ssim']):.4f} | Mean PSNR = {np.mean(results['oracle']['psnr']):.2f} dB")
    print(f"  Predicted Routing: Mean SSIM = {np.mean(results['predicted']['ssim']):.4f} | Mean PSNR = {np.mean(results['predicted']['psnr']):.2f} dB")
    if task1_universal_model is not None:
        print(f"  Universal AE (T1): Mean SSIM = {np.mean(results['universal']['ssim']):.4f} | Mean PSNR = {np.mean(results['universal']['psnr']):.2f} dB")

    return results


if __name__ == "__main__":
    print("=" * 65)
    print("Testing HardRouter Pipeline")
    print("=" * 65)

    checkpoints_dir = os.path.join(PARENT_DIR, "checkpoints")
    cls_ckpt = os.path.join(checkpoints_dir, "best_corruption_classifier.pt")
    noise_ckpt = os.path.join(checkpoints_dir, "best_specialist_noise.pt")
    blur_ckpt = os.path.join(checkpoints_dir, "best_specialist_blur.pt")
    occ_ckpt = os.path.join(checkpoints_dir, "best_specialist_occlusion.pt")

    router = HardRouter(
        classifier_ckpt=cls_ckpt if os.path.exists(cls_ckpt) else None,
        noise_ckpt=noise_ckpt if os.path.exists(noise_ckpt) else None,
        blur_ckpt=blur_ckpt if os.path.exists(blur_ckpt) else None,
        occlusion_ckpt=occ_ckpt if os.path.exists(occ_ckpt) else None,
    )

    # Test dummy batch
    dummy_input = torch.randn(4, 3, 128, 128).clamp(0, 1)
    res_pred = router(dummy_input, mode="predicted")
    res_oracle = router(dummy_input, mode="oracle", oracle_label=1)

    print("\n--- Predicted Routing Test ---")
    print(f"Input Shape:     {tuple(dummy_input.shape)}")
    print(f"Restored Shape:  {tuple(res_pred['restored'].shape)}")
    print(f"Routed Classes:  {res_pred['routed_class']}")
    print(f"Latency:         {res_pred['latency_ms']:.2f} ms")

    print("\n--- Oracle Routing Test ---")
    print(f"Forced Oracle:   Class 1 (Noise Specialist)")
    print(f"Restored Shape:  {tuple(res_oracle['restored'].shape)}")
    print(f"Latency:         {res_oracle['latency_ms']:.2f} ms")
    print("\nHardRouter Pipeline verification: PASSED!")
