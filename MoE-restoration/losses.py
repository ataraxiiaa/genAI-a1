"""
Task 3: Joint Loss Function and Routing Balance Regularizer
Loss Formulation:
    L_joint = lambda_rec * L_L1 + lambda_ssim * (1 - SSIM) + lambda_ce * L_ce + lambda_bal * L_balance

Components:
1. L1 Reconstruction Loss: Pixel-level absolute difference ||x_hat - y||_1
2. Structural Similarity Loss: 1 - SSIM(x_hat, y), preserving edges, textures, and luminance
3. Corruption Classification Loss: Cross-entropy on gating logits G(x) vs known corruption label
4. Routing Balance Regularizer: L_balance = sum_{k=1}^4 (w_bar_k - 0.25)^2, where w_bar_k is
   the batch-average routing weight assigned to branch k. Prevents single-expert collapse.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class BalanceRegularizer(nn.Module):
    """
    Differentiable MoE Routing Balance Regularizer.
    Penalizes deviations of the batch-average routing distribution from uniform (1 / K = 0.25).
    Formula:
        L_balance = sum_{k=1}^K ( (1/B * sum_{i=1}^B w_{i,k}) - 1/K )^2
    """

    def __init__(self, num_branches: int = 4):
        super().__init__()
        self.num_branches = num_branches
        self.target_prob = 1.0 / num_branches

    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            routing_weights: (B, K) tensor of softmax routing probabilities
        Returns:
            Scalar balance loss
        """
        # Batch mean for each branch: (K,)
        w_bar = torch.mean(routing_weights, dim=0)
        # Sum of squared deviations from target uniform allocation
        balance_loss = torch.sum((w_bar - self.target_prob) ** 2)
        return balance_loss


class JointSoftMoELoss(nn.Module):
    """
    Four-component joint loss for end-to-end training of the Soft MoE model.
    """

    def __init__(
        self,
        lambda_rec: float = 0.8,
        lambda_ssim: float = 0.2,
        lambda_ce: float = 0.1,
        lambda_bal: float = 0.01,
        num_branches: int = 4,
    ):
        super().__init__()
        self.lambda_rec = float(lambda_rec)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_ce = float(lambda_ce)
        self.lambda_bal = float(lambda_bal)

        self.l1_loss = nn.L1Loss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.balance_regularizer = BalanceRegularizer(num_branches=num_branches)

    def forward(
        self,
        reconstruction: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total joint loss and individual breakdown.
        Args:
            reconstruction: (B, 3, H, W) restored output
            targets: (B, 3, H, W) clean ground-truth images
            logits: (B, 4) gating network unnormalized logits
            labels: (B,) integer corruption class labels [0, 1, 2, 3]
            routing_weights: (B, 4) softmax routing weights
        Returns:
            total_loss: scalar tensor for .backward()
            loss_dict: dictionary of float values for metrics logging
        """
        # 1. Reconstruction Losses
        loss_l1 = self.l1_loss(reconstruction, targets)
        ssim_val = self.ssim_metric(reconstruction, targets)
        loss_ssim = 1.0 - ssim_val

        # 2. Gating Classification Loss
        loss_ce = self.ce_loss(logits, labels)

        # 3. Routing Balance Regularizer
        loss_bal = self.balance_regularizer(routing_weights)

        # 4. Joint Weighted Sum
        total_loss = (
            self.lambda_rec * loss_l1
            + self.lambda_ssim * loss_ssim
            + self.lambda_ce * loss_ce
            + self.lambda_bal * loss_bal
        )

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "loss_l1": loss_l1.item(),
            "loss_ssim": loss_ssim.item(),
            "ssim": ssim_val.item(),
            "loss_ce": loss_ce.item(),
            "loss_bal": loss_bal.item(),
        }

        return total_loss, loss_breakdown


if __name__ == "__main__":
    print("=" * 65)
    print("Testing JointSoftMoELoss Boilerplate")
    print("=" * 65)

    criterion = JointSoftMoELoss(
        lambda_rec=0.8,
        lambda_ssim=0.2,
        lambda_ce=0.1,
        lambda_bal=0.01,
    )

    B = 8
    dummy_pred = torch.rand(B, 3, 128, 128, requires_grad=True)
    dummy_target = torch.rand(B, 3, 128, 128)
    dummy_logits = torch.randn(B, 4, requires_grad=True)
    dummy_labels = torch.randint(0, 4, (B,))
    dummy_weights = F.softmax(dummy_logits, dim=-1)

    loss, breakdown = criterion(dummy_pred, dummy_target, dummy_logits, dummy_labels, dummy_weights)

    print("Computed Loss Breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:12s}: {v:.5f}")

    # Test backward pass
    loss.backward()
    assert dummy_pred.grad is not None, "Gradient missing on reconstruction"
    assert dummy_logits.grad is not None, "Gradient missing on logits"
    print("\nDifferentiable backward pass verified successfully!")
    print("=" * 65)
