"""
Soft Mixture-of-Experts (MoE) Restoration Package
Task 3 of Generative AI Assignment #1

Contains:
- SoftMoERestoration: Differentiable 4-branch MoE architecture with Gating Network
- JointSoftMoELoss: 4-component differentiable objective (L1, SSIM, Cross-Entropy, Routing Balance)
- Excel Tracker: Multi-sheet training logger for Task 3
"""

from .soft_moe import SoftMoERestoration
from .losses import JointSoftMoELoss, BalanceRegularizer

__all__ = [
    "SoftMoERestoration",
    "JointSoftMoELoss",
    "BalanceRegularizer",
]
