"""
Task 4: Loss Functions for Face-to-Sketch Conditional GAN (Pix2Pix)
Loss Formulation:
1. Generator Loss:
   L_G = L_adv + lambda_L1 * L_L1(y, G(x, s))
   - Adversarial Loss (L_adv): Binary cross-entropy encouraging generated sketches
     to fool the PatchGAN discriminator into predicting 'real' (1s).
   - Reconstruction Loss (L_L1): Pixel-level absolute error ||y - G(x, s)||_1 preserving
     anatomical structure, facial identity, and alignment.
   - Initial weighting: lambda_L1 = 100.0 (investigated via Optuna).

2. Discriminator Loss:
   L_D = 0.5 * (L_real + L_fake)
   - L_real: BCE loss on real (photo, real_sketch, style) pairs targeting 1s.
   - L_fake: BCE loss on fake (photo, generated_sketch, style) pairs targeting 0s.
   - Scaled by 0.5 to balance discriminator vs generator gradient updates.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure


class GeneratorLoss(nn.Module):
    """
    Combined Generator objective for conditional image-to-image synthesis:
    L_G = BCEWithLogits(D(x, G(x, s), s), 1) + lambda_L1 * ||y - G(x, s)||_1
    """

    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def set_lambda_l1(self, lambda_l1: float) -> None:
        """Update L1 reconstruction weight (e.g. during Optuna trials)."""
        self.lambda_l1 = float(lambda_l1)

    def forward(
        self,
        d_fake_logits: torch.Tensor,
        generated_sketch: torch.Tensor,
        real_sketch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes combined generator loss.
        Args:
            d_fake_logits: (B, 1, 14, 14) PatchGAN logits on generated pairs
            generated_sketch: (B, 3, 128, 128) synthesized sketch G(x, s) in [-1, 1]
            real_sketch: (B, 3, 128, 128) ground-truth sketch y in [-1, 1]
        Returns:
            total_loss: scalar tensor for G backpropagation
            loss_breakdown: dictionary containing individual float loss values
        """
        # 1. Adversarial Loss: Generator wants discriminator to output 1 (Real) for its fakes
        real_targets = torch.ones_like(d_fake_logits)
        loss_adv = self.bce_loss(d_fake_logits, real_targets)

        # 2. Reconstruction L1 Loss: Pixel-wise correspondence between synthetic and target sketch
        loss_l1 = self.l1_loss(generated_sketch, real_sketch)

        # 3. Total Weighted Generator Loss
        total_loss = loss_adv + self.lambda_l1 * loss_l1

        breakdown = {
            "g_total": total_loss.item(),
            "g_adv": loss_adv.item(),
            "g_l1": loss_l1.item(),
            "g_weighted_l1": (self.lambda_l1 * loss_l1).item(),
        }

        return total_loss, breakdown


class DiscriminatorLoss(nn.Module):
    """
    PatchGAN Discriminator objective:
    L_D = 0.5 * ( BCEWithLogits(D_real, 1) + BCEWithLogits(D_fake, 0) )
    Supports optional one-sided label smoothing (e.g. real targets = 0.9) to stabilize GAN training.
    """

    def __init__(self, label_smoothing: float = 1.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        d_real_logits: torch.Tensor,
        d_fake_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes discriminator realism classification loss.
        Args:
            d_real_logits: (B, 1, 14, 14) PatchGAN logits on real pairs D(x, y, s)
            d_fake_logits: (B, 1, 14, 14) PatchGAN logits on fake pairs D(x, G(x, s).detach(), s)
        Returns:
            total_loss: scalar tensor for D backpropagation
            loss_breakdown: dictionary containing individual float loss values
        """
        # 1. Real pair loss: target = label_smoothing (1.0 or 0.95)
        real_targets = torch.full_like(d_real_logits, self.label_smoothing)
        loss_real = self.bce_loss(d_real_logits, real_targets)

        # 2. Fake pair loss: target = 0.0
        fake_targets = torch.zeros_like(d_fake_logits)
        loss_fake = self.bce_loss(d_fake_logits, fake_targets)

        # 3. Average loss
        total_loss = 0.5 * (loss_real + loss_fake)

        breakdown = {
            "d_total": total_loss.item(),
            "d_real": loss_real.item(),
            "d_fake": loss_fake.item(),
        }

        return total_loss, breakdown


def compute_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Calculates L1 error, SSIM, and PSNR between generated sketches and ground-truth.
    Tensors expected in range [-1.0, 1.0], converted internally to [0.0, 1.0] for standard metrics.
    Returns:
        (l1_error, ssim_score, psnr_db)
    """
    # Denormalize [-1, 1] -> [0, 1]
    gen_01 = (generated * 0.5 + 0.5).clamp(0.0, 1.0)
    tgt_01 = (target * 0.5 + 0.5).clamp(0.0, 1.0)

    # 1. L1 Error
    l1_val = torch.mean(torch.abs(gen_01 - tgt_01)).item()

    # 2. SSIM
    ssim_val = structural_similarity_index_measure(gen_01, tgt_01, data_range=1.0).item()

    # 3. PSNR
    mse = torch.mean((gen_01 - tgt_01) ** 2).item()
    if mse <= 1e-10:
        psnr_val = 50.0
    else:
        psnr_val = 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    return l1_val, ssim_val, psnr_val


if __name__ == "__main__":
    print("=" * 65)
    print("Testing Face-to-Sketch Conditional GAN Losses")
    print("=" * 65)

    B = 4
    dummy_d_real = torch.randn(B, 1, 14, 14, requires_grad=True)
    dummy_d_fake = torch.randn(B, 1, 14, 14, requires_grad=True)
    dummy_gen_sketch = torch.randn(B, 3, 128, 128, requires_grad=True)
    dummy_real_sketch = torch.randn(B, 3, 128, 128)

    # Test Discriminator Loss
    d_criterion = DiscriminatorLoss(label_smoothing=0.95)
    d_loss, d_breakdown = d_criterion(dummy_d_real, dummy_d_fake)
    d_loss.backward(retain_graph=True)
    print("Discriminator Loss Breakdown:")
    for k, v in d_breakdown.items():
        print(f"  {k:15s}: {v:.4f}")
    assert dummy_d_real.grad is not None and dummy_d_fake.grad is not None, "D gradient missing!"

    # Test Generator Loss
    g_criterion = GeneratorLoss(lambda_l1=100.0)
    g_loss, g_breakdown = g_criterion(dummy_d_fake, dummy_gen_sketch, dummy_real_sketch)
    g_loss.backward()
    print("\nGenerator Loss Breakdown:")
    for k, v in g_breakdown.items():
        print(f"  {k:15s}: {v:.4f}")
    assert dummy_gen_sketch.grad is not None, "G gradient missing on output sketch!"

    # Test Metrics computation
    l1_m, ssim_m, psnr_m = compute_metrics(dummy_gen_sketch.detach(), dummy_real_sketch)
    print(f"\nSample Evaluation Metrics:")
    print(f"  L1 Error:  {l1_m:.4f}")
    print(f"  SSIM:      {ssim_m:.4f}")
    print(f"  PSNR:      {psnr_m:.2f} dB")

    print("\nAll Face-to-Sketch GAN Loss modules verified successfully!")
    print("=" * 65)
