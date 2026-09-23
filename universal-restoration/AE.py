"""
Universal Denoising Autoencoder Boilerplate
Task 1: Oxford-IIIT Pet Dataset Restoration (128x128 RGB)

Requirements from Assignment:
- Convolutional encoder that progressively reduces spatial dimensions (128 -> ... -> bottleneck)
- Genuine compressed latent representation (meaningful bottleneck)
- Convolutional decoder that reconstructs the clean RGB image (bottleneck -> ... -> 128)
- No unrestricted skip connections (to force learning robust compressed features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from torchmetrics.functional.image import structural_similarity_index_measure


class ConvBlock(nn.Module):
    """
    Convolutional building block: Conv2d -> BatchNorm -> LeakyReLU -> optional Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    Transposed convolutional block: ConvTranspose2d -> BatchNorm -> ReLU -> optional Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1 if stride == 2 else 0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    Convolutional Encoder for 128x128 RGB images.
    
    Progressively reduces spatial dimensions while expanding channels:
      Stage 1: (3, 128, 128)          -> (base_channels, 64, 64)
      Stage 2: (base_channels, 64, 64) -> (base_channels * 2, 32, 32)
      Stage 3: (base_channels * 2, 32, 32) -> (base_channels * 4, 16, 16)
      Stage 4: (base_channels * 4, 16, 16) -> (base_channels * 8, 8, 8)
      Bottleneck: (base_channels * 8, 8, 8) -> (latent_dim, 8, 8)
    
    Includes configurable dropout rate for Optuna hyperparameter tuning.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 256,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # 4 downsampling stages (128 -> 64 -> 32 -> 16 -> 8)
        self.stage1 = ConvBlock(in_channels, base_channels, stride=2, dropout=0.0)
        self.stage2 = ConvBlock(base_channels, base_channels * 2, stride=2, dropout=0.0)
        self.stage3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, dropout=dropout_rate)
        self.stage4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2, dropout=dropout_rate)

        # Bottleneck compression layer to lock into the exact latent_dim
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 128, 128)
        Returns:
            z: Latent bottleneck tensor of shape (B, latent_dim, 8, 8)
        """
        x = self.stage1(x)       # (B, base_channels, 64, 64)
        x = self.stage2(x)       # (B, base_channels * 2, 32, 32)
        x = self.stage3(x)       # (B, base_channels * 4, 16, 16)
        x = self.stage4(x)       # (B, base_channels * 8, 8, 8)
        z = self.bottleneck(x)   # (B, latent_dim, 8, 8)
        return z


class Decoder(nn.Module):
    """
    Convolutional Decoder for reconstructing 128x128 RGB images.
    
    Progressively increases spatial dimensions back to original resolution:
      Unbottleneck: (latent_dim, 8, 8)           -> (base_channels * 8, 8, 8)
      Stage 1:      (base_channels * 8, 8, 8)    -> (base_channels * 4, 16, 16)
      Stage 2:      (base_channels * 4, 16, 16)  -> (base_channels * 2, 32, 32)
      Stage 3:      (base_channels * 2, 32, 32)  -> (base_channels, 64, 64)
      Stage 4:      (base_channels, 64, 64)      -> (base_channels, 128, 128)
      Final Conv:   (base_channels, 128, 128)    -> (out_channels, 128, 128) with Sigmoid
    """
    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 32,
        out_channels: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Unbottleneck layer: projects latent vector back to base_channels * 8 feature map
        self.unbottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # 4 upsampling stages (8 -> 16 -> 32 -> 64 -> 128)
        self.stage1 = DeconvBlock(base_channels * 8, base_channels * 4, stride=2, dropout=dropout_rate)
        self.stage2 = DeconvBlock(base_channels * 4, base_channels * 2, stride=2, dropout=dropout_rate)
        self.stage3 = DeconvBlock(base_channels * 2, base_channels, stride=2, dropout=0.0)
        self.stage4 = DeconvBlock(base_channels, base_channels, stride=2, dropout=0.0)

        # Final projection to RGB image with Sigmoid to keep output strictly in [0.0, 1.0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent bottleneck tensor of shape (B, latent_dim, 8, 8)
        Returns:
            x_hat: Reconstructed RGB image tensor of shape (B, 3, 128, 128) in [0.0, 1.0]
        """
        x = self.unbottleneck(z)  # (B, base_channels * 8, 8, 8)
        x = self.stage1(x)        # (B, base_channels * 4, 16, 16)
        x = self.stage2(x)        # (B, base_channels * 2, 32, 32)
        x = self.stage3(x)        # (B, base_channels, 64, 64)
        x = self.stage4(x)        # (B, base_channels, 128, 128)
        x_hat = self.final_conv(x)# (B, 3, 128, 128)
        return x_hat


class DenoisingAutoencoder(nn.Module):
    """
    Universal Denoising Autoencoder.
    Takes corrupted image x_tilde -> Encoder -> Bottleneck (z) -> Decoder -> Restored image x_hat
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 256,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            out_channels=in_channels,
            dropout_rate=dropout_rate,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns compressed latent features."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs RGB image from latent features."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: corrupted input -> restored output."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# ==============================================================================
# Loss Function & Backward Pass Boilerplate
# ==============================================================================

class UDAELoss(nn.Module):
    """
    Loss function for Task 1: Universal Denoising Autoencoder.
    Formula:
        Loss = alpha * L1_loss + (1 - alpha) * (1 - SSIM)
    """
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        restored: torch.Tensor,
        clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            restored: Model prediction (B, 3, 128, 128) in [0.0, 1.0]
            clean: Ground-truth clean image (B, 3, 128, 128) in [0.0, 1.0]
        Returns:
            total_loss: Differentiable scalar loss for backpropagation
            metrics: Dictionary of numerical metrics for logging
        """
        l1 = F.l1_loss(restored, clean)
        ssim_val = structural_similarity_index_measure(restored, clean, data_range=1.0)
        ssim_loss = 1.0 - ssim_val

        total_loss = self.alpha * l1 + (1.0 - self.alpha) * ssim_loss

        metrics = {
            "total_loss": total_loss.item(),
            "l1_loss": l1.item(),
            "ssim": ssim_val.item(),
        }
        return total_loss, metrics


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: UDAELoss,
    corrupted: torch.Tensor,
    clean: torch.Tensor,
) -> Dict[str, float]:
    """
    Performs one complete training iteration:
      1. Zero out old gradients
      2. Forward pass (compute prediction)
      3. Compute loss (L1 + SSIM)
      4. Backward pass (backpropagate gradients through the network)
      5. Optimizer step (update model weights)
    """
    model.train()

    # Step 1: Clear gradients from previous batch
    optimizer.zero_grad()

    # Step 2: Forward pass
    restored = model(corrupted)

    # Step 3: Compute combined loss
    loss, metrics = criterion(restored, clean)

    # Step 4: Backward pass (Auto-differentiation calculates gradients)
    loss.backward()

    # Step 5: Update weights using calculated gradients
    optimizer.step()

    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Autoencoder on device: {device}")

    # 1. Initialize Model, Loss Function, and Optimizer
    model = DenoisingAutoencoder(base_channels=32, latent_dim=256, dropout_rate=0.1).to(device)
    criterion = UDAELoss(alpha=0.8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. Create Dummy Batch (Batch size 4, 3 channels, 128x128)
    dummy_corrupted = torch.rand(4, 3, 128, 128, device=device)
    dummy_clean = torch.rand(4, 3, 128, 128, device=device)

    # 3. Test Forward Pass
    print("\n--- 1. Testing Forward Pass ---")
    with torch.no_grad():
        latent = model.encode(dummy_corrupted)
        reconstruction = model(dummy_corrupted)
    print(f"Input shape:        {dummy_corrupted.shape}")
    print(f"Bottleneck shape:   {latent.shape}")
    print(f"Output shape:       {reconstruction.shape}")

    # 4. Test Backward Pass (Full Training Step)
    print("\n--- 2. Testing Backward Pass & Optimization Step ---")
    metrics = train_step(model, optimizer, criterion, dummy_corrupted, dummy_clean)
    print(f"Backward Pass executed successfully!")
    print(f"Batch Loss: {metrics['total_loss']:.4f} (L1: {metrics['l1_loss']:.4f}, SSIM: {metrics['ssim']:.4f})")
    print("\nAutoencoder forward & backward pipeline successfully verified!")
