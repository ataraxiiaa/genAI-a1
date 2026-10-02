"""
Task 4: Conditional GAN Face-to-Sketch Synthesis Package
Contains:
- FS2KDataset & FS2K Dataloaders: Paired Photo-Sketch dataset with stratified validation split
- UNetGenerator: Style-conditioned U-Net Generator with skip connections
- PatchGANDiscriminator: 70x70 PatchGAN Discriminator with style embedding conditioning
- GANLoss: Adversarial BCE and L1 reconstruction loss objectives
"""

from .fs2k_loader import FS2KDataset, get_fs2k_dataloaders
from .models import UNetGenerator, PatchGANDiscriminator, StyleEmbedding
from .losses import GeneratorLoss, DiscriminatorLoss

__all__ = [
    "FS2KDataset",
    "get_fs2k_dataloaders",
    "UNetGenerator",
    "PatchGANDiscriminator",
    "StyleEmbedding",
    "GeneratorLoss",
    "DiscriminatorLoss",
]
