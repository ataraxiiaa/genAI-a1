# Task 4: Face-to-Sketch Conditional GAN — Architectural & Mathematical Learning Guide

This document explains the core deep learning concepts, architectural mechanisms, and mathematical formulations powering the style-conditioned Face-to-Sketch synthesis system.

---

## Table of Contents
1. [Spatial Broadcasting Explained](#1-spatial-broadcasting-explained)
2. [U-Net Encoder-Decoder with Skip Connections](#2-u-net-encoder-decoder-with-skip-connections)
3. [PatchGAN Discriminator & 70x70 Local Receptive Field](#3-patchgan-discriminator--70x70-local-receptive-field)
4. [Loss Formulations: Adversarial Minimax + L1 Reconstruction](#4-loss-formulations-adversarial-minimax--l1-reconstruction)
5. [Learned Categorical Style Embeddings](#5-learned-categorical-style-embeddings)
6. [Paired Dataset Alignment & Identical Augmentations](#6-paired-dataset-alignment--identical-augmentations)

---

## 1. Spatial Broadcasting Explained

### The Simple Analogy
Imagine you have a colored stamp (a style code) that is a 1x1 dot. But your canvas is a big 128x128 grid.
If you only stamp that color on a single corner pixel, the rest of the canvas will not know what style you want.
**Spatial Broadcasting** is like painting that exact color evenly across every single coordinate of the 128x128 grid so that every pixel location receives the instruction: *"Render this area in Style S."*

### Why We Need It
- Photographs and sketches are 2D spatial feature maps with dimensions `(Batch, Channels, Height, Width)`.
- A style label (e.g., `0` for Shaded Pencil, `1` for Clean Contour, `2` for Graphic Comic) is a single scalar integer per image.
- Even when converted to a vector of length $E$ (embedding dimension), it only has shape `(B, E)`.
- Standard 2D convolutional layers cannot directly concatenate a 2D vector `(B, E)` with a 4D spatial tensor `(B, 3, 128, 128)`.

### How PyTorch Implements It
```python
# 1. Look up learned continuous vector:
#    Input: style_idx with shape (B,)
#    Output: emb with shape (B, E)
emb = self.style_embedding(style_idx)

# 2. Reshape into 4D tensor with spatial dimensions 1 x 1:
#    Output: shape (B, E, 1, 1)
emb_4d = emb.view(B, E, 1, 1)

# 3. Spatially expand across the full resolution (H=128, W=128):
#    Memory-efficient view expansion (zero memory copy until used)
#    Output: shape (B, E, 128, 128)
emb_spatial = emb_4d.expand(-1, -1, H, W)

# 4. Concatenate along channel dimension (dim=1):
#    photo: (B, 3, 128, 128) + emb_spatial: (B, E, 128, 128)
#    x: (B, 3 + E, 128, 128)
x = torch.cat([photo, emb_spatial], dim=1)
```

### Usage in the Generator & Discriminator
1. **In the Generator (`UNetGenerator`):**
   - The photo and style channels are combined right at the input: `(B, 3 + E, 128, 128)`.
   - Every receptive field across every convolutional filter has immediate access to both the raw facial pixels and the style conditioning signal from the very first layer.
2. **In the Discriminator (`PatchGANDiscriminator`):**
   - The discriminator receives the photo `(3 channels)`, the candidate sketch `(3 channels)`, and the spatial style embedding `(E channels)`.
   - Total input: `6 + E channels`.
   - The discriminator checks whether the candidate sketch looks like a real sketch of the given photo **in that specific style**. If the generator draws a clean contour when the style requested was shaded pencil, the discriminator detects the mismatch and penalizes it.

---

## 2. U-Net Encoder-Decoder with Skip Connections

### The Bottleneck Problem in Standard Autoencoders
In a standard encoder-decoder without skip connections:
1. The encoder progressively downsamples the image from $128 \times 128 \to 64 \times 64 \to \dots \to 4 \times 4$.
2. Spatial resolution is heavily compressed into an abstract semantic bottleneck.
3. Fine high-frequency details (individual pencil strokes, eyelash hairs, sharp nose contours) are lost because $4 \times 4$ cannot store pixel-level coordinates.
4. When the decoder reconstructs the sketch, it produces blurry, smudged lines.

### The U-Net Solution
U-Net adds **skip connections** linking each encoder stage directly to its corresponding decoder stage:

```
Input (128x128) -----------------------------------------> Final Output (128x128)
    \                                                          ^
  Down 1 (64x64) -------- [Skip Connection 1] -------> Up 4 (64x64)
      \                                                    ^
    Down 2 (32x32) ------ [Skip Connection 2] ------> Up 3 (32x32)
        \                                              ^
      Down 3 (16x16) ---- [Skip Connection 3] ----> Up 2 (16x16)
          \                                        ^
        Down 4 (8x8) ---- [Skip Connection 4] -> Up 1 (8x8)
            \                                      ^
           Down 5 (4x4) [Bottleneck Feature] ------+
```

### What Each Path Does
- **Downsampling Path (Encoder):** Extracts abstract high-level facial geometry (face shape, eye positions, head orientation) using strided $4 \times 4$ convolutions (stride 2, padding 1) with LeakyReLU activations.
- **Skip Connections:** Relay crisp, exact spatial coordinates directly across the network.
- **Upsampling Path (Decoder):** Uses Transposed Convolutions ($4 \times 4$, stride 2, padding 1) to double resolution. At each step, it concatenates the decoded features with the high-resolution skip features from the encoder, followed by ReLU activations and optional Dropout to prevent overfitting.
- **Final Activation:** Uses `nn.Tanh()`, bounding all output pixel intensities strictly to $[-1.0, 1.0]$.

---

## 3. PatchGAN Discriminator & 70x70 Local Receptive Field

### Global Discriminator vs. PatchGAN
- A **Standard GAN Discriminator** outputs a single scalar ($0$ to $1$) for the entire $128 \times 128$ image. This forces the discriminator to evaluate global structure, often ignoring blurry local patches or repeated stroke artifacts.
- A **PatchGAN Discriminator** penalizes structure at the scale of local image patches. It outputs a matrix of logits (e.g., $14 \times 14$ or $16 \times 16$).
- Each element in the output matrix represents whether a specific **$70 \times 70$ receptive field patch** in the input image appears real or synthetic.

### Advantages of PatchGAN
1. **Fewer Parameters & Faster Training:** By operating locally with a fixed receptive field, the discriminator requires fewer parameters than a full-image global discriminator.
2. **Sharp High-Frequency Focus:** PatchGAN acts as a learned high-frequency texture and style discriminator, forcing the generator to create crisp strokes and realistic shading.
3. **Full Image Realism by Convolution:** The average of all patch losses forms the total adversarial loss, ensuring that every local region across the entire face is scrutinized.

---

## 4. Loss Formulations: Adversarial Minimax + L1 Reconstruction

The conditional GAN is trained using a two-part objective combining an adversarial minimax loss and an L1 pixel-level reconstruction loss.

### 1. Discriminator Loss ($L_D$)
The discriminator learns to classify real photo-sketch pairs as `1.0` (with one-sided label smoothing, e.g. `0.95`) and generated photo-sketch pairs as `0.0`:
$$L_D = \frac{1}{2} \mathbb{E}\left[ (D(x, y, s) - 1)^2 \right] + \frac{1}{2} \mathbb{E}\left[ (D(x, G(x, s), s) - 0)^2 \right]$$
Using Least-Squares GAN (LSGAN) or Binary Cross-Entropy (BCEWithLogitsLoss) ensures smooth gradients and prevents gradient vanishing.

### 2. Generator Loss ($L_G$)
The generator aims to fool the discriminator while simultaneously matching the ground truth sketch pixel-for-pixel:
$$L_G = L_{\text{adv}} + \lambda_{L1} \cdot L_{L1}$$
Where:
- **Adversarial Loss:** $L_{\text{adv}} = \mathbb{E}\left[ -\log D(x, G(x, s), s) \right]$ (encourages realistic, sharp sketch textures that fool the discriminator).
- **L1 Reconstruction Loss:** $L_{L1} = \mathbb{E}\left[ \|y - G(x, s)\|_1 \right]$ (enforces structural fidelity and accurate alignment with facial features).
- **Balancing Hyperparameter ($\lambda_{L1}$):** Typically set between $50.0$ and $150.0$ (default $100.0$). L1 alone causes blurry outputs; GAN alone causes hallucinations. Combined, they produce photo-accurate, crisp sketches.

---

## 5. Learned Categorical Style Embeddings

Instead of one-hot vectors, which treat all styles as orthogonal and unlearnable, we employ `nn.Embedding(num_styles=3, embedding_dim=16)`:
- Each style index ($s \in \{0, 1, 2\}$) maps to a continuous, differentiable vector in $\mathbb{R}^{16}$.
- During training, backpropagation updates the embedding weights.
- The network learns rich, distributed representations of pencil pressure, contour line weight, and hatching density characteristic of each of the 3 FS2K styles.

---

## 6. Paired Dataset Alignment & Identical Augmentations

### The Spatial Alignment Rule
In paired image-to-image translation (Pix2Pix), every pixel in photograph $x$ corresponds to an identical anatomical feature in target sketch $y$.
If we apply data augmentations independently:
- A horizontal flip applied only to the photo would pair a left-facing face with a right-facing sketch.
- The generator would receive contradictory supervisory signals, causing severe ghosting artifacts and blurred facial symmetry.

### Solution: Synchronized Transformation
Both photograph and sketch must undergo the **identical geometric transformation**:
```python
if torch.rand(1).item() > 0.5:
    photo = TF.hflip(photo)
    sketch = TF.hflip(sketch)

# Identical rotation angle
angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())
photo = TF.rotate(photo, angle)
sketch = TF.rotate(sketch, angle)
```
Color jittering (brightness, contrast) is applied **only to the input photograph**, training the generator to remain invariant to lighting variations while maintaining consistent sketch rendering.
