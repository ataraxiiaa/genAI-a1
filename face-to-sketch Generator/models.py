"""
Task 4: Conditional GAN Architecture (Style-Conditioned Pix2Pix for Face-to-Sketch)
Models:
1. StyleEmbedding:
   - Learned categorical embedding for the 3 FS2K sketch styles
   - Maps style index in {0, 1, 2} to dense vector of dimension embed_dim
2. UNetGenerator:
   - U-Net Encoder-Decoder with skip connections & spatial style condition injection
   - Input: x (B, 3, 128, 128) facial photo + s (B,) style condition
   - Output: y_hat (B, 3, 128, 128) synthesized sketch with Tanh activation [-1.0, 1.0]
3. PatchGANDiscriminator:
   - 70x70 receptive field PatchGAN
   - Input: photo x + sketch y (real or fake) + style embedding s
   - Output: Local patch realism logits (B, 1, 14, 14)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STYLES = 3


def weights_init_normal(m: nn.Module) -> None:
    """
    Standard Pix2Pix weight initialization:
    Normal distribution with mean=0.0, std=0.02 for Convs, and mean=1.0, std=0.02 for BatchNorm.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class StyleEmbedding(nn.Module):
    """
    Learned categorical embedding module for FS2K sketch styles.
    Maps discrete style index {0, 1, 2} to a dense latent vector of dimension embed_dim.
    """

    def __init__(self, num_styles: int = NUM_STYLES, embed_dim: int = 16):
        super().__init__()
        self.num_styles = num_styles
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_styles, embed_dim)

    def forward(self, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_idx: (B,) long tensor with class indices [0, 1, 2]
        Returns:
            emb: (B, embed_dim) continuous style embedding
        """
        return self.embedding(style_idx)


class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling stage:
    Conv2d(kernel=4, stride=2, padding=1) -> BatchNorm2d -> LeakyReLU(0.2) -> Dropout2d (optional)
    Halves spatial dimensions (H, W) -> (H/2, W/2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling stage:
    ConvTranspose2d(kernel=4, stride=2, padding=1) -> BatchNorm2d -> ReLU -> Dropout2d (optional)
    Doubles spatial dimensions (H, W) -> (2H, 2W).
    Concatenates skip connection from the corresponding downsampling stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.block(x)
        if skip is not None:
            out = torch.cat([out, skip], dim=1)
        return out


class UNetGenerator(nn.Module):
    """
    Style-Conditioned U-Net Generator with Skip Connections.
    Resolution pipeline:
      128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 (Bottleneck)
      4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 (Output)
    Style condition s is embedded into dimension embed_dim and spatially broadcasted
    across the entire input image, establishing global style conditioning from stage 1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 16,
        num_styles: int = NUM_STYLES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.style_embedding = StyleEmbedding(num_styles=num_styles, embed_dim=embed_dim)

        # Input channels: 3 (RGB photo) + embed_dim (spatial style embedding)
        total_in = in_channels + embed_dim

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.down1 = UNetDownBlock(total_in, base_channels, normalize=False)          # 128 -> 64
        self.down2 = UNetDownBlock(base_channels, base_channels * 2, normalize=True)  # 64 -> 32
        self.down3 = UNetDownBlock(base_channels * 2, base_channels * 4, normalize=True, dropout=dropout) # 32 -> 16
        self.down4 = UNetDownBlock(base_channels * 4, base_channels * 8, normalize=True, dropout=dropout) # 16 -> 8
        self.down5 = UNetDownBlock(base_channels * 8, base_channels * 8, normalize=True, dropout=dropout) # 8 -> 4

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        # up1: 4 -> 8. In: 8C, Out: 8C. Concat with down4 (8C) -> 16C
        self.up1 = UNetUpBlock(base_channels * 8, base_channels * 8, dropout=dropout)
        # up2: 8 -> 16. In: 16C, Out: 4C. Concat with down3 (4C) -> 8C
        self.up2 = UNetUpBlock(base_channels * 16, base_channels * 4, dropout=dropout)
        # up3: 16 -> 32. In: 8C, Out: 2C. Concat with down2 (2C) -> 4C
        self.up3 = UNetUpBlock(base_channels * 8, base_channels * 2)
        # up4: 32 -> 64. In: 4C, Out: C. Concat with down1 (C) -> 2C
        self.up4 = UNetUpBlock(base_channels * 4, base_channels)

        # Final Layer: 64 -> 128. In: 2C, Out: 3. Tanh activation ensures range [-1.0, 1.0]
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(weights_init_normal)

    def forward(self, photo: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes sketch y_hat from photo x conditioned on style s.
        Args:
            photo: (B, 3, 128, 128) facial photograph in range [-1.0, 1.0]
            style_idx: (B,) integer tensor with style category in {0, 1, 2}
        Returns:
            generated_sketch: (B, 3, 128, 128) generated sketch in range [-1.0, 1.0]
        """
        B, _, H, W = photo.shape

        # 1. Project style index to learned continuous embedding
        emb = self.style_embedding(style_idx)  # (B, embed_dim)
        # 2. Spatially broadcast across (H, W)
        emb_spatial = emb.view(B, self.embed_dim, 1, 1).expand(-1, -1, H, W)

        # 3. Concatenate photo and style condition along channel axis
        x = torch.cat([photo, emb_spatial], dim=1)  # (B, 3 + embed_dim, 128, 128)

        # 4. Encoder pass with skip feature retention
        d1 = self.down1(x)    # (B, C, 64, 64)
        d2 = self.down2(d1)   # (B, 2C, 32, 32)
        d3 = self.down3(d2)   # (B, 4C, 16, 16)
        d4 = self.down4(d3)   # (B, 8C, 8, 8)
        d5 = self.down5(d4)   # (B, 8C, 4, 4)

        # 5. Decoder pass with skip connections
        u1 = self.up1(d5, d4)  # (B, 16C, 8, 8)
        u2 = self.up2(u1, d3)  # (B, 8C, 16, 16)
        u3 = self.up3(u2, d2)  # (B, 4C, 32, 32)
        u4 = self.up4(u3, d1)  # (B, 2C, 64, 64)

        # 6. Final output projection
        out = self.final_up(u4)  # (B, 3, 128, 128)
        return out


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator with Style Conditioning.
    Evaluates whether local overlapping patches of a (photo, sketch) pair are real or generated.
    Conditioned on the learned style embedding.
    Input Channels:
      - 3 (photo) + 3 (sketch) + embed_dim (style condition) = 6 + embed_dim
    Output:
      - (B, 1, 14, 14) unnormalized patch realism logits
    """

    def __init__(
        self,
        in_channels: int = 6,  # 3 (photo) + 3 (sketch)
        base_channels: int = 64,
        embed_dim: int = 16,
        num_styles: int = NUM_STYLES,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.style_embedding = StyleEmbedding(num_styles=num_styles, embed_dim=embed_dim)

        total_in = in_channels + embed_dim

        self.model = nn.Sequential(
            # Stage 1: (6 + embed_dim) -> C (128x128 -> 64x64)
            nn.Conv2d(total_in, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: C -> 2C (64x64 -> 32x32)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: 2C -> 4C (32x32 -> 16x16)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: 4C -> 8C (16x16 -> 15x15, stride 1)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: 8C -> 1 (15x15 -> 14x14 patch classification)
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        self.apply(weights_init_normal)

    def forward(
        self,
        photo: torch.Tensor,
        sketch: torch.Tensor,
        style_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classifies realism of photo-sketch pairs under given style condition.
        Args:
            photo: (B, 3, 128, 128) facial photograph
            sketch: (B, 3, 128, 128) real or generated sketch
            style_idx: (B,) integer style category
        Returns:
            patch_logits: (B, 1, 14, 14) unnormalized realism score map
        """
        B, _, H, W = photo.shape
        emb = self.style_embedding(style_idx)
        emb_spatial = emb.view(B, self.embed_dim, 1, 1).expand(-1, -1, H, W)

        # Concatenate photo, sketch, and style condition
        x = torch.cat([photo, sketch, emb_spatial], dim=1)
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Face-to-Sketch Conditional GAN Architectures")
    print("=" * 70)

    B = 4
    dummy_photo = torch.randn(B, 3, 128, 128)
    dummy_styles = torch.randint(0, 3, (B,))

    netG = UNetGenerator(base_channels=64, embed_dim=16, dropout=0.1)
    netD = PatchGANDiscriminator(base_channels=64, embed_dim=16)

    print(f"Generator Parameters:     {count_parameters(netG):,}")
    print(f"Discriminator Parameters: {count_parameters(netD):,}")

    # Forward pass Generator
    fake_sketch = netG(dummy_photo, dummy_styles)
    print(f"\nGenerator Output Shape:   {tuple(fake_sketch.shape)}")
    print(f"Generator Value Range:    [{fake_sketch.min().item():.3f}, {fake_sketch.max().item():.3f}]")

    # Forward pass Discriminator (Real vs Fake pairs)
    dummy_real_sketch = torch.randn(B, 3, 128, 128)
    d_real = netD(dummy_photo, dummy_real_sketch, dummy_styles)
    d_fake = netD(dummy_photo, fake_sketch.detach(), dummy_styles)

    print(f"Discriminator Output Shape: {tuple(d_real.shape)}")
    assert fake_sketch.shape == (B, 3, 128, 128), "Generator output shape mismatch!"
    assert d_real.shape == (B, 1, 14, 14), "Discriminator patch grid shape mismatch!"

    print("\nAll Face-to-Sketch models verified successfully!")
    print("=" * 70)
