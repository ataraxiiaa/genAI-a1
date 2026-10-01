"""
Task 2: Lightweight 4-Class Corruption Classifier
Architecture:
- Custom 4-stage Convolutional Feature Extractor
- Adaptive Global Average Pooling (spatial invariance)
- Dense Classification Head outputting 4 class logits:
    0: Clean (Identity bypass)
    1: Noise (Gaussian / Salt & Pepper)
    2: Blur (Gaussian Blur)
    3: Box Occlusion (Inpainting)
"""

import torch
import torch.nn as nn
from typing import List, Dict

# Canonical 4-class labels mapping exactly aligned with loader.py
CLASS_NAMES: List[str] = [
    "clean",
    "salt_and_pepper",
    "gaussian_blur",
    "rectangular_occlusion",
]
SHORT_NAMES: List[str] = ["clean", "noise", "blur", "occlusion"]
NUM_CLASSES: int = len(CLASS_NAMES)



class ConvBlock(nn.Module):
    """
    Standard feature extraction block:
    Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d (optional) -> Dropout (optional)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CorruptionClassifier(nn.Module):
    """
    Lightweight CNN Classifier for identifying input corruption type.
    Input:  (B, 3, 128, 128) RGB image tensor in range [0, 1]
    Output: (B, 4) unnormalized class logits
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Feature Extractor Backbone: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.features = nn.Sequential(
            # Stage 1: 3 -> 32, size 128 -> 64
            ConvBlock(in_channels, base_channels, pool=True, dropout_rate=0.0),
            # Stage 2: 32 -> 64, size 64 -> 32
            ConvBlock(base_channels, base_channels * 2, pool=True, dropout_rate=dropout_rate),
            # Stage 3: 64 -> 128, size 32 -> 16
            ConvBlock(base_channels * 2, base_channels * 4, pool=True, dropout_rate=dropout_rate),
            # Stage 4: 128 -> 256, size 16 -> 8
            ConvBlock(base_channels * 4, base_channels * 8, pool=True, dropout_rate=dropout_rate),
        )

        # Global Average Pooling collapses 8x8 spatial grid into 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        in_features = base_channels * 8
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw unnormalized class logits (B, 4)."""
        feat = self.features(x)
        pooled = self.gap(feat)
        logits = self.classifier_head(pooled)
        return logits

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the predicted class index (0 to 3) for routing."""
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax confidence distribution across the 4 classes."""
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Lightweight CorruptionClassifier Architecture")
    print("=" * 60)

    # Initialize model
    model = CorruptionClassifier(in_channels=3, num_classes=4, base_channels=32, dropout_rate=0.1)
    params = count_parameters(model)
    print(f"Total Trainable Parameters: {params:,} (~{params * 4 / (1024 * 1024):.2f} MB float32)")

    # Test with dummy batch
    dummy_input = torch.randn(4, 3, 128, 128)
    logits = model(dummy_input)
    preds = model.predict_class(dummy_input)
    probs = model.predict_probabilities(dummy_input)

    print(f"Input Shape:         {tuple(dummy_input.shape)}")
    print(f"Logits Shape:        {tuple(logits.shape)}")
    print(f"Predictions Shape:   {tuple(preds.shape)} -> Predicted indices: {preds.tolist()}")
    print(f"Probabilities Shape: {tuple(probs.shape)}")
    print("Sample Probs (Batch 0):", [f"{CLASS_NAMES[i]}: {p:.3f}" for i, p in enumerate(probs[0].tolist())])
    print("Classifier architecture check: PASSED!")
