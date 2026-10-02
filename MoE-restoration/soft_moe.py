"""
Task 3: Differentiable Soft Mixture-of-Experts (MoE) Restoration Model
Architecture:
- Gating Network: Pre-trained 4-Class Corruption Classifier producing routing logits G(x)
- Routing Weights: w = softmax(G(x) / tau) across 4 branches:
    Branch 0: Clean Image Bypass (Identity)
    Branch 1: Salt-and-Pepper Noise Specialist Autoencoder
    Branch 2: Gaussian Blur Specialist Autoencoder
    Branch 3: Rectangular Occlusion Inpainting Specialist Autoencoder
- Reconstruction:
    x_hat = w[0]*x + w[1]*A_salt(x) + w[2]*A_blur(x) + w[3]*A_occlusion(x)
- Temperature Parameter tau: Controls sharpness (tau -> 0) vs. uniformity (tau -> inf)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root and subpackages are in sys.path for imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPECIALIST_DIR = os.path.join(WORKSPACE_ROOT, "specialist-autoencoders")
for p in [WORKSPACE_ROOT, SPECIALIST_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from classifier import CorruptionClassifier
from specialists import (
    NoiseSpecialistAE,
    BlurSpecialistAE,
    OcclusionSpecialistAE,
    IdentityBypass,
)

BRANCH_NAMES: List[str] = [
    "clean_identity",
    "noise_specialist",
    "blur_specialist",
    "occlusion_specialist",
]
BRANCH_SHORT_NAMES: List[str] = ["clean", "noise", "blur", "occlusion"]
NUM_BRANCHES: int = len(BRANCH_NAMES)


class SoftMoERestoration(nn.Module):
    """
    Differentiable Soft Mixture-of-Experts (MoE) restoration architecture.
    Combines 3 specialist autoencoders and an identity bypass weighted by
    a temperature-scaled gating network.
    """

    def __init__(
        self,
        gate_base_channels: int = 16,
        gate_dropout: float = 0.05,
        specialist_base_channels: int = 48,
        specialist_latent_dim: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.num_branches = NUM_BRANCHES

        # 1. Gating Network (from Task 2 classifier architecture)
        self.gating_network = CorruptionClassifier(
            in_channels=3,
            num_classes=self.num_branches,
            base_channels=gate_base_channels,
            dropout_rate=gate_dropout,
        )

        # 2. Expert Branches
        self.identity_bypass = IdentityBypass()
        self.noise_expert = NoiseSpecialistAE(
            base_channels=specialist_base_channels,
            latent_dim=specialist_latent_dim,
        )
        self.blur_expert = BlurSpecialistAE(
            base_channels=specialist_base_channels,
            latent_dim=specialist_latent_dim,
        )
        self.occlusion_expert = OcclusionSpecialistAE(
            base_channels=specialist_base_channels,
            latent_dim=specialist_latent_dim,
        )

    def set_temperature(self, tau: float) -> None:
        """Dynamically update softmax routing temperature."""
        if tau <= 0:
            raise ValueError(f"Temperature tau must be strictly positive, got {tau}")
        self.temperature = float(tau)

    def freeze_experts(self) -> None:
        """Freeze all 3 specialist autoencoders for Stage 1 Warm-up."""
        for expert in [self.noise_expert, self.blur_expert, self.occlusion_expert]:
            for param in expert.parameters():
                param.requires_grad = False
        print("[SoftMoE] Experts frozen for Stage 1 Gate Warm-up.")

    def unfreeze_experts(self) -> None:
        """Unfreeze all 3 specialist autoencoders for Stage 2 Joint Fine-Tuning."""
        for expert in [self.noise_expert, self.blur_expert, self.occlusion_expert]:
            for param in expert.parameters():
                param.requires_grad = True
        print("[SoftMoE] Experts unfrozen for Stage 2 Joint Fine-Tuning.")

    def freeze_gate(self) -> None:
        """Freeze the gating network."""
        for param in self.gating_network.parameters():
            param.requires_grad = False
        print("[SoftMoE] Gating network frozen.")

    def unfreeze_gate(self) -> None:
        """Unfreeze the gating network."""
        for param in self.gating_network.parameters():
            param.requires_grad = True
        print("[SoftMoE] Gating network unfrozen.")

    def load_pretrained_weights(
        self,
        classifier_path: Optional[str] = None,
        noise_path: Optional[str] = None,
        blur_path: Optional[str] = None,
        occlusion_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, bool]:
        """
        Load weights from pre-trained Task 2 models.
        """
        status = {}

        # 1. Gate / Classifier
        if classifier_path and os.path.exists(classifier_path):
            ckpt = torch.load(classifier_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.gating_network.load_state_dict(state_dict)
            status["gate"] = True
            print(f"[SoftMoE] Loaded pre-trained Gating Network from {classifier_path}")
        else:
            status["gate"] = False
            print(f"[SoftMoE] Warning: Gating Network weights not found at {classifier_path}")

        # 2. Noise Specialist
        if noise_path and os.path.exists(noise_path):
            ckpt = torch.load(noise_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.noise_expert.load_state_dict(state_dict)
            status["noise"] = True
            print(f"[SoftMoE] Loaded pre-trained Noise Expert from {noise_path}")
        else:
            status["noise"] = False
            print(f"[SoftMoE] Warning: Noise Expert weights not found at {noise_path}")

        # 3. Blur Specialist
        if blur_path and os.path.exists(blur_path):
            ckpt = torch.load(blur_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.blur_expert.load_state_dict(state_dict)
            status["blur"] = True
            print(f"[SoftMoE] Loaded pre-trained Blur Expert from {blur_path}")
        else:
            status["blur"] = False
            print(f"[SoftMoE] Warning: Blur Expert weights not found at {blur_path}")

        # 4. Occlusion Specialist
        if occlusion_path and os.path.exists(occlusion_path):
            ckpt = torch.load(occlusion_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.occlusion_expert.load_state_dict(state_dict)
            status["occlusion"] = True
            print(f"[SoftMoE] Loaded pre-trained Occlusion Expert from {occlusion_path}")
        else:
            status["occlusion"] = False
            print(f"[SoftMoE] Warning: Occlusion Expert weights not found at {occlusion_path}")

        return status

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
        return_intermediates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Differentiable forward pass.
        Args:
            x: Input batch of images (B, 3, 128, 128) in range [0, 1]
            temperature: Optional override for softmax temperature tau
            return_intermediates: Whether to return individual branch outputs
        Returns:
            reconstruction: (B, 3, 128, 128) final blended restoration
            w: (B, 4) soft routing weights for [clean, noise, blur, occlusion]
            logits: (B, 4) raw gating logits G(x)
            (Optional) intermediates: dict with branch outputs
        """
        tau = float(temperature) if temperature is not None else self.temperature

        # 1. Gating Network forward pass
        logits = self.gating_network(x)  # (B, 4)
        w = F.softmax(logits / tau, dim=-1)  # (B, 4)

        # 2. Compute individual branch restorations
        x_clean = self.identity_bypass(x)         # (B, 3, 128, 128)
        x_noise = self.noise_expert(x)           # (B, 3, 128, 128)
        x_blur = self.blur_expert(x)             # (B, 3, 128, 128)
        x_occlusion = self.occlusion_expert(x)   # (B, 3, 128, 128)

        # 3. Continuous Differentiable Linear Combination
        # w has shape (B, 4), expand each weight to (B, 1, 1, 1)
        w_clean = w[:, 0:1, None, None]
        w_noise = w[:, 1:2, None, None]
        w_blur = w[:, 2:3, None, None]
        w_occ = w[:, 3:4, None, None]

        reconstruction = (
            w_clean * x_clean +
            w_noise * x_noise +
            w_blur * x_blur +
            w_occ * x_occlusion
        )
        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)

        if return_intermediates:
            intermediates = {
                "clean": x_clean,
                "noise": x_noise,
                "blur": x_blur,
                "occlusion": x_occlusion,
            }
            return reconstruction, w, logits, intermediates

        return reconstruction, w, logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SoftMoERestoration Architecture Boilerplate")
    print("=" * 70)

    moe = SoftMoERestoration(
        gate_base_channels=16,
        gate_dropout=0.05,
        specialist_base_channels=48,
        specialist_latent_dim=256,
        temperature=1.0,
    )

    tot, train = count_parameters(moe)
    print(f"Total Parameters:     {tot:,}")
    print(f"Trainable Parameters: {train:,}")

    # Test warm-up freezing
    moe.freeze_experts()
    tot, train_warmup = count_parameters(moe)
    print(f"Trainable after freeze (Warm-up): {train_warmup:,} (Gate only)")

    moe.unfreeze_experts()
    tot, train_joint = count_parameters(moe)
    print(f"Trainable after unfreeze (Joint): {train_joint:,} (Full System)")

    # Test Forward Pass
    dummy_input = torch.rand(4, 3, 128, 128)
    recon, weights, logits, inter = moe(dummy_input, return_intermediates=True)

    print("\nForward Pass Verification:")
    print(f"  Input Shape:         {tuple(dummy_input.shape)}")
    print(f"  Reconstruction:      {tuple(recon.shape)}")
    print(f"  Routing Weights:     {tuple(weights.shape)}")
    print(f"  Logits Shape:        {tuple(logits.shape)}")
    print(f"  Weight Sum per sample: {weights.sum(dim=-1).tolist()} (Should be ~1.0)")
    print(f"  Sample 0 weights:    {weights[0].tolist()}")

    assert recon.shape == dummy_input.shape, "Reconstruction shape mismatch"
    assert weights.shape == (4, 4), "Weights shape mismatch"
    assert logits.shape == (4, 4), "Logits shape mismatch"
    print("\nBoilerplate SoftMoERestoration successfully verified!")
    print("=" * 70)
