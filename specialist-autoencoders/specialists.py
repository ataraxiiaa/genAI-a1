"""
Task 2: Specialist Autoencoders Architecture Boilerplate
Defines the 3 Specialist Autoencoders + Clean Identity Bypass:
1. NoiseSpecialistAE: Denoising Autoencoder specialized for Salt-and-Pepper noise
2. BlurSpecialistAE: Deblurring Autoencoder specialized for Gaussian blur
3. OcclusionSpecialistAE: Inpainting Autoencoder specialized for rectangular box occlusions
4. IdentityBypass: Zero-compute bypass for clean uncorrupted images
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Downsampling convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU -> Dropout2d"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Upsampling deconvolutional block: ConvTranspose2d -> BatchNorm2d -> ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseSpecialistAE(nn.Module):
    """
    Base Convolutional Autoencoder Architecture for Specialists.
    Input:  (B, 3, 128, 128) corrupted image in [0, 1]
    Output: (B, 3, 128, 128) reconstructed clean image in [0, 1]
    
    Structure:
    - 4 Downsampling Stages: 128 -> 64 -> 32 -> 16 -> 8
    - Latent Bottleneck: (B, latent_dim, 8, 8)
    - 4 Upsampling Stages: 8 -> 16 -> 32 -> 64 -> 128
    - Output Head: 1x1 Conv + Sigmoid
    """
    def __init__(
        self,
        base_channels: int = 64,
        latent_dim: int = 256,
        dropout_rate: float = 0.05,
        name: str = "BaseSpecialist",
    ):
        super().__init__()
        self.name = name
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        # Encoder: 128x128 -> 8x8
        self.encoder = nn.Sequential(
            # Stage 1: 3 -> C (128x128 -> 64x64)
            ConvBlock(3, base_channels, dropout_rate=0.0),
            # Stage 2: C -> 2C (64x64 -> 32x32)
            ConvBlock(base_channels, base_channels * 2, dropout_rate=dropout_rate),
            # Stage 3: 2C -> 4C (32x32 -> 16x16)
            ConvBlock(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate),
            # Stage 4: 4C -> latent_dim (16x16 -> 8x8)
            ConvBlock(base_channels * 4, latent_dim, dropout_rate=dropout_rate),
        )

        # Decoder: 8x8 -> 128x128
        self.decoder = nn.Sequential(
            # Stage 1: latent_dim -> 4C (8x8 -> 16x16)
            DeconvBlock(latent_dim, base_channels * 4),
            # Stage 2: 4C -> 2C (16x16 -> 32x32)
            DeconvBlock(base_channels * 4, base_channels * 2),
            # Stage 3: 2C -> C (32x32 -> 64x64)
            DeconvBlock(base_channels * 2, base_channels),
            # Stage 4: C -> C/2 (64x64 -> 128x128)
            DeconvBlock(base_channels, base_channels // 2),
            # Output Layer: C/2 -> 3 (128x128) + Sigmoid
            nn.Conv2d(base_channels // 2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


# ==============================================================================
# Specialist 1: Denoising Autoencoder
# ==============================================================================
class NoiseSpecialistAE(BaseSpecialistAE):
    """
    Specialist Autoencoder trained EXCLUSIVELY on Salt-and-Pepper noise.
    Learns non-linear spatial filtering to eliminate impulse pixels without blurring high frequencies.
    """
    def __init__(self, base_channels: int = 64, latent_dim: int = 256, dropout_rate: float = 0.05):
        super().__init__(
            base_channels=base_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            name="NoiseSpecialist",
        )


# ==============================================================================
# Specialist 2: Deblurring Autoencoder
# ==============================================================================
class BlurSpecialistAE(BaseSpecialistAE):
    """
    Specialist Autoencoder trained EXCLUSIVELY on Gaussian blur corruptions.
    Learns inverse-filter deconvolution to restore sharp texture details and edges.
    """
    def __init__(self, base_channels: int = 64, latent_dim: int = 256, dropout_rate: float = 0.05):
        super().__init__(
            base_channels=base_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            name="BlurSpecialist",
        )


# ==============================================================================
# Specialist 3: Inpainting Autoencoder
# ==============================================================================
class OcclusionSpecialistAE(BaseSpecialistAE):
    """
    Specialist Autoencoder trained EXCLUSIVELY on rectangular box occlusions.
    Learns semantic contextual filling to plausibly hallucinate missing anatomical regions.
    """
    def __init__(self, base_channels: int = 64, latent_dim: int = 256, dropout_rate: float = 0.05):
        super().__init__(
            base_channels=base_channels,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            name="OcclusionSpecialist",
        )


# ==============================================================================
# Clean Bypass: Identity Module
# ==============================================================================
class IdentityBypass(nn.Module):
    """
    Clean corruption bypass module.
    When classifier detects class 0 (clean), returns the input unchanged (0 ms compute).
    """
    def __init__(self):
        super().__init__()
        self.name = "IdentityBypass"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_specialist(corruption_type: str, base_channels: int = 64, latent_dim: int = 256) -> nn.Module:
    """Factory helper to instantiate a specialist by name."""
    type_norm = corruption_type.lower().strip()
    if type_norm in ["noise", "salt_and_pepper"]:
        return NoiseSpecialistAE(base_channels=base_channels, latent_dim=latent_dim)
    elif type_norm in ["blur", "gaussian_blur"]:
        return BlurSpecialistAE(base_channels=base_channels, latent_dim=latent_dim)
    elif type_norm in ["occlusion", "rectangular_occlusion", "box_occlusion"]:
        return OcclusionSpecialistAE(base_channels=base_channels, latent_dim=latent_dim)
    elif type_norm in ["clean", "identity"]:
        return IdentityBypass()
    else:
        raise ValueError(f"Unknown specialist corruption type: '{corruption_type}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 65)
    print("Testing 3 Specialist Autoencoders Architecture Boilerplate")
    print("=" * 65)

    specialists = {
        "Noise Specialist (Salt & Pepper)": NoiseSpecialistAE(),
        "Blur Specialist (Gaussian Blur)": BlurSpecialistAE(),
        "Occlusion Specialist (Inpainting)": OcclusionSpecialistAE(),
        "Clean Bypass (Identity)": IdentityBypass(),
    }

    dummy_input = torch.randn(2, 3, 128, 128)

    for name, model in specialists.items():
        params = count_parameters(model)
        output = model(dummy_input)
        print(f"\n[{name}]")
        print(f"  Trainable Parameters: {params:,}")
        print(f"  Input Shape:          {tuple(dummy_input.shape)}")
        print(f"  Output Shape:         {tuple(output.shape)}")
        print(f"  Output Range:         [{output.min().item():.3f}, {output.max().item():.3f}]")
        assert output.shape == dummy_input.shape, f"Shape mismatch in {name}!"

    print("\n" + "=" * 65)
    print("All Specialist Boilerplates verified and PASSED!")
    print("=" * 65)
