- test-severity-configs are basically how corrupted the image is
- gaussian-blur also has the area that should be blurred (kernel size) and how much blur (sigma)

- we create a deterministic validation/testing setup we use same seed so the corruption type is same for the image, the blurness,area mask coordinates and severity they remain consistent, we do this cuz for e.g if we have epoch 1 doing a tiny blur easy test but if in epoch 2 test got harder we wouldn't know if our model imporved/worsened or the test just changed

- we have a validation and testing manifesting generation process and save these manifests so that the corruptions are deterministic we can load these manifests every epoch to evaluate our model under same conditions

- for the convolution block we used leakyRLU as activation function and for deconvolutaion we used ReLU
- in encoder if we use standard RLU the corrupted image that may have negative nubmers will become zero, if a neuron only outputs negative values we will have dying ReLU problem
- in decoder our job is reconstruction and standard ReLU will create sparsity it can turn off features that shouldn't be in picture
- Other activation functions that could've been used
  1. GELU (Gaussian Error Linear Unit) smoothing out negative inputs
  2. PReLU learns the optimal slop unlike leakyRLU
  3. SwiLU (Swish-GELU Hybrid) smooth non-linearities

- Optimizer step adamW (udpating weights) gives momentum and adaptive learning rate

Mean Absolute Error (L1) : Measures average pixel difference (smooth)
Structural Similarity (SSIM): Measures contrast, luminance, structure (perceptual)
Peak Signal-to-Noise Ratio (PSNR): Logarithmic scale of reconstruction quality

- ONNX is not python its a c++ inference engine we can use the .onnx file in other programming languages and platfroms like it's basically a model file format for deployment
  since it cannot understand python and see whats going on we use **TRACING** send a package with same dimensions so the onnx engine can trace/record the operations performed on it and then create a graph and see what steps it went through
